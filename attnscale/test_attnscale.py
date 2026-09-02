"""Unit tests for attnscale.  Run:  python attnscale/test_attnscale.py"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attnscale import (FractionalKeyNorm, KeyScale, QKNorm, ScaledAttention,  # noqa: E402
                       initial_logit_std, scale_key_projection_, suggest_key_scale)

torch.manual_seed(0)


def _harness_key_math(k4, gain, alpha, ema, c=8.0):
    """The exact key-side math of experiments/2026-08-31 & 2026-09-01 run.py (knorm_dynk arm)."""
    B, T, H, D = k4.shape
    rk = k4.pow(2).mean(-1).add(1e-12).sqrt()
    kn = k4 * torch.rsqrt(k4.pow(2).mean(-1, keepdim=True) + 1e-6)
    ratio = (rk / ema.view(1, 1, -1)).clamp(1.0 / c, c)
    tau = torch.exp(alpha.view(1, 1, -1) * ratio.log())
    kn = kn * tau.unsqueeze(-1)
    return (kn.reshape(B, T, H * D) * gain).view(B, T, H, D)


def test_matches_lab_harness_math():
    B, T, H, D = 2, 7, 4, 8
    k = torch.randn(B, T, H, D) * 3
    m = FractionalKeyNorm(H, D).eval()
    with torch.no_grad():
        m.alpha.copy_(torch.tensor([0.2, 0.5, 0.9, 1.3]))
        m.gain.copy_(torch.randn(H * D).abs() + 0.5)
        m.rms_ema.copy_(torch.tensor([1.0, 2.0, 0.5, 3.0]))
        m.ema_ready.fill_(1.0)
    ref = _harness_key_math(k, m.gain, m.alpha, m.rms_ema)
    assert torch.allclose(m(k), ref, atol=1e-6, rtol=1e-6)


def test_alpha_zero_is_key_only_rmsnorm():
    H, D = 3, 16
    k = torch.randn(4, 5, H, D)
    m = FractionalKeyNorm(H, D, alpha_init=0.0).eval()
    ref = QKNorm(H, D)
    with torch.no_grad():
        ref.gain.copy_(m.gain)
    assert torch.allclose(m(k), ref(k), atol=1e-6)


def test_alpha_one_is_running_scale_only():
    H, D = 2, 8
    m = FractionalKeyNorm(H, D, alpha_init=1.0).eval()
    with torch.no_grad():
        m.rms_ema.copy_(torch.tensor([2.0, 0.5]))
        m.ema_ready.fill_(1.0)
    k = torch.randn(3, 6, H, D)       # rms ~1, ratio inside the clamp
    ref = k / m.rms_ema.view(1, 1, H, 1)
    assert torch.allclose(m(k), ref, atol=1e-5)


def test_gradient_flows_through_magnitude():
    """Radial gradient is zero for alpha=0 (pure norm) and nonzero for alpha=1."""
    H, D = 1, 8
    for alpha, expect_radial in ((0.0, False), (1.0, True)):
        m = FractionalKeyNorm(H, D, alpha_init=alpha).eval()
        with torch.no_grad():
            m.ema_ready.fill_(1.0)
        k = torch.randn(1, 1, H, D, requires_grad=True)
        out = m(k)
        (out * out).sum().backward()          # loss = ||k_hat||^2
        radial = (k.grad * k).sum().abs().item()
        if expect_radial:
            assert radial > 1e-3, radial
        else:
            assert radial < 1e-3, radial   # eps in the RMS makes the norm only ~scale-invariant


def test_ema_updates_only_in_train():
    H, D = 2, 4
    m = FractionalKeyNorm(H, D)
    k = torch.randn(8, 16, H, D) * 5
    assert float(m.ema_ready) == 0.0
    m.train()
    m(k)
    e1 = m.rms_ema.clone()
    assert float(m.ema_ready) == 1.0
    m(k * 2)
    e2 = m.rms_ema.clone()
    assert not torch.allclose(e1, e2)
    m.eval()
    m(k * 10)
    assert torch.allclose(m.rms_ema, e2)


def test_layout_bhtd_equivalent():
    H, D = 4, 8
    a = FractionalKeyNorm(H, D, layout="bthd").eval()
    b = FractionalKeyNorm(H, D, layout="bhtd").eval()
    b.load_state_dict(a.state_dict())
    k = torch.randn(2, 9, H, D)
    assert torch.allclose(a(k), b(k.transpose(1, 2)).transpose(1, 2), atol=1e-6)


def test_nanogpt_attention_shapes_and_backward():
    cfg = SimpleNamespace(n_embd=64, n_head=8, block_size=32, dropout=0.0, bias=False)
    for key in ("none", "kscale", "qknorm", "knorm", "fkn"):
        att = ScaledAttention(cfg, key=key)
        x = torch.randn(3, 20, 64, requires_grad=True)
        y = att(x)
        assert y.shape == x.shape
        y.pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        if key == "fkn":
            assert att.k_op.alpha.grad is not None


def test_bf16_autocast_cpu():
    cfg = SimpleNamespace(n_embd=32, n_head=4, block_size=16, dropout=0.0, bias=False)
    att = ScaledAttention(cfg, key="fkn")
    x = torch.randn(2, 10, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = att(x)
    assert torch.isfinite(y.float()).all()


def test_deterministic():
    H, D = 2, 8
    k = torch.randn(2, 5, H, D)
    a = FractionalKeyNorm(H, D).eval()
    assert torch.equal(a(k), a(k))


def test_keyscale_is_a_constant_multiplier():
    H, D = 4, 8
    k = torch.randn(2, 5, H, D)
    m = KeyScale(H, c=8.0)
    assert torch.allclose(m(k), k * 8.0, atol=1e-6)
    assert sum(p.numel() for p in m.parameters()) == 0        # zero parameters when fixed
    per_head = KeyScale(H, c=[1.0, 2.0, 4.0, 8.0])
    out = per_head(k)
    for h, c in enumerate([1.0, 2.0, 4.0, 8.0]):
        assert torch.allclose(out[:, :, h], k[:, :, h] * c, atol=1e-6)


def test_keyscale_layouts_and_learnable():
    H, D = 3, 8
    k = torch.randn(2, 7, H, D)
    a, b = KeyScale(H, c=3.0, layout="bthd"), KeyScale(H, c=3.0, layout="bhtd")
    assert torch.allclose(a(k), b(k.transpose(1, 2)).transpose(1, 2), atol=1e-6)
    learn = KeyScale(H, c=2.0, learnable=True)
    learn(k).pow(2).sum().backward()
    assert learn.log_c.grad is not None and torch.isfinite(learn.log_c.grad).all()


def test_scale_key_projection_matches_keyscale_in_forward():
    """Folding into a fused [q;k;v] weight changes the keys exactly as KeyScale does."""
    torch.manual_seed(0)
    d, H = 32, 4
    lin = torch.nn.Linear(d, 3 * d, bias=False)
    x = torch.randn(2, 6, d)
    q0, k0, v0 = lin(x).split(d, dim=2)
    scale_key_projection_(lin, 4.0, d_model=d)
    q1, k1, v1 = lin(x).split(d, dim=2)
    assert torch.allclose(q0, q1, atol=1e-6) and torch.allclose(v0, v1, atol=1e-6)
    assert torch.allclose(k1, k0 * 4.0, atol=1e-5)


def test_diagnostics_report_the_scale():
    torch.manual_seed(0)
    B, T, H, D = 4, 16, 4, 8
    k = torch.randn(B, T, H, D) * 0.2
    q = torch.randn(B, T, H, D) * 0.2
    s_small = initial_logit_std(q, k, D)
    s_big = initial_logit_std(q, k * 8, D)
    assert s_big > 5 * s_small                       # the diagnostic tracks the scale
    c = suggest_key_scale(k)
    assert c.shape == (H,)
    rescaled = k * c.view(1, 1, H, 1)
    assert abs(float(rescaled.pow(2).mean(-1).sqrt().mean()) - 1.0) < 0.05   # ~unit RMS


if __name__ == "__main__":
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} tests passed")
