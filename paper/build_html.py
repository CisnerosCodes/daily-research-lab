"""Build paper/index.html from paper/paper.md and paper/figures/*.png.

Usage:  python paper/build_html.py
- Markdown -> HTML (python-markdown: tables, fenced code, toc).
- <!-- FIGn --> markers become theme-aware figure blocks (light + dark PNG, base64-inlined).
- Remaining <!-- ... --> placeholders are rendered as visible "pending" notes so nothing is silently blank.
"""
import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

FIGURES = {
    "FIG1": ("fig1_phase_diagram", "Figure 1. Validation loss across the iso-parameter head split for the five core arms "
             "(600 steps, three paired seeds; small dots are individual seeds). FKN is the orange line."),
    "FIG2": ("fig2_delta_vs_baseline", "Figure 2. The same runs as a change relative to the unnormalized baseline at each head width. "
             "Negative is better; the FKN bars carry their values."),
    "FIG3": ("fig3_alpha_vs_headdim", "Figure 3. The learned exponent alpha as a function of head width. alpha = 0 is pure key-only "
             "normalization; alpha = 1 keeps the full per-token magnitude relative to the running scale."),
    "FIG4": ("fig4_cliff_decomposition", "Figure 4. Six key-side arms at head_dim 4 on paired inits (registry 2026-08-31). The arm that "
             "restores the magnitude value with its gradient severed pays the full cliff; the arm that reopens the gradient lands below baseline."),
    "FIG5": ("fig5_thread_timeline", "Figure 5. The best configuration at head_dim 4 found on each night of the thread, relative to no normalization."),
    "FIG6": ("fig6_ptb_transfer", "Figure 6. Second corpus: character-level Penn Treebank at the cliff, the QK-norm optimum and the wide split "
             "(600 steps, three paired seeds)."),
    "FIG7": ("fig7_longer_training", "Figure 7. Three times longer training (1800 steps) at head_dim 4 and 64, three paired seeds."),
    "FIG8": ("fig8_mqar_recipe", "Figure 8. Left: only a dense per-channel gate leaves the MQAR guessing plateau at identical state size "
             "(registry 2026-07-28). Right: escape step against the gradient-noise scale lr/B; batch 256 at 4x learning rate escapes at step 300 "
             "in every seed (registry 2026-09-01)."),
    "FIG10": ("fig10_adaptive_vs_static", "Figure 10. Is the win adaptive? Keys divided by a per-head running scale at three momenta, "
              "the same scale frozen at its first-batch value, and the identical trick on the query side (registry 2026-09-01_kscale-adaptive-vs-static)."),
    "FIG9": ("fig9_loop_test_time_compute", "Figure 9. Test-time depth extrapolation on prefix parity (registry 2026-07-25): a weight-tied loop "
             "trained with a stochastic depth schedule keeps solving harder instances past its trained depth; fixed-depth training and untied depth do not."),
}


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def figure_block(key: str) -> str:
    stem, caption = FIGURES[key]
    light, dark = FIG / f"{stem}_light.png", FIG / f"{stem}_dark.png"
    if not light.exists():
        return f'<div class="pending">Figure {key[3:]} is generated once its experiment finishes.</div>'
    return (f'<figure class="fig" id="{stem}">'
            f'<img class="only-light" src="{b64(light)}" alt="{caption}">'
            f'<img class="only-dark" src="{b64(dark)}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


CSS = """
:root {
  color-scheme: light;
  --ground: #f7f8fa; --panel: #ffffff; --ink: #101216; --ink-2: #4b525c; --ink-3: #7b828c;
  --rule: #dfe3e8; --rule-2: #cbd1d8; --accent: #d95926; --accent-soft: #fbe9e1; --code: #eef1f4;
  --link: #1f5fa8;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #17191d; --panel: #1d2025; --ink: #eceef2; --ink-2: #b4bac4; --ink-3: #868d97;
    --rule: #2c3036; --rule-2: #3a3f46; --accent: #eb6834; --accent-soft: #3a261d; --code: #23272d;
    --link: #7cb0ec;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #17191d; --panel: #1d2025; --ink: #eceef2; --ink-2: #b4bac4; --ink-3: #868d97;
  --rule: #2c3036; --rule-2: #3a3f46; --accent: #eb6834; --accent-soft: #3a261d; --code: #23272d;
  --link: #7cb0ec;
}
.only-dark { display: none; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .only-light { display: none; } :root:not([data-theme="light"]) .only-dark { display: block; } }
:root[data-theme="dark"] .only-light { display: none; } :root[data-theme="dark"] .only-dark { display: block; }
:root[data-theme="light"] .only-light { display: block; } :root[data-theme="light"] .only-dark { display: none; }

html { background: var(--ground); }
body { margin: 0; background: var(--ground); color: var(--ink); font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
       font-size: 17px; line-height: 1.6; -webkit-font-smoothing: antialiased; }
.wrap { display: grid; grid-template-columns: 1fr; gap: 0; max-width: 1180px; margin: 0 auto; padding: 0 20px 80px; }
@media (min-width: 1040px) { .wrap { grid-template-columns: 250px minmax(0, 760px); gap: 48px; } }
nav.toc { display: none; }
@media (min-width: 1040px) {
  nav.toc { display: block; position: sticky; top: 24px; align-self: start; max-height: calc(100vh - 48px); overflow-y: auto;
            padding: 20px 0 20px 0; font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 13px; line-height: 1.45; }
  nav.toc .eyebrow { color: var(--ink-3); text-transform: uppercase; letter-spacing: .08em; font-size: 11px; margin-bottom: 10px; }
  nav.toc ul { list-style: none; margin: 0; padding: 0; }
  nav.toc li { margin: 0 0 6px 0; }
  nav.toc ul ul { padding-left: 12px; margin-top: 4px; }
  nav.toc ul ul li { color: var(--ink-3); margin-bottom: 3px; }
  nav.toc a { color: var(--ink-2); text-decoration: none; }
  nav.toc a:hover { color: var(--accent); }
}
main { min-width: 0; padding-top: 36px; }
header.title { border-bottom: 1px solid var(--rule); padding-bottom: 22px; margin-bottom: 26px; }
.kicker { font-family: "IBM Plex Sans", system-ui, sans-serif; text-transform: uppercase; letter-spacing: .1em; font-size: 11.5px; color: var(--accent); margin-bottom: 14px; }
h1 { font-family: "IBM Plex Sans", system-ui, sans-serif; font-weight: 600; font-size: clamp(28px, 3.4vw, 38px); line-height: 1.15; letter-spacing: -0.01em; margin: 0 0 14px; text-wrap: balance; }
.subtitle { font-size: 19px; color: var(--ink-2); margin: 0 0 16px; text-wrap: balance; }
.meta { font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 13px; color: var(--ink-3); }
h2 { font-family: "IBM Plex Sans", system-ui, sans-serif; font-weight: 600; font-size: 24px; letter-spacing: -0.01em; margin: 44px 0 12px; padding-top: 10px; text-wrap: balance; }
h3 { font-family: "IBM Plex Sans", system-ui, sans-serif; font-weight: 600; font-size: 17px; margin: 28px 0 8px; color: var(--ink); }
p { margin: 0 0 14px; max-width: 70ch; }
li { max-width: 70ch; }
a { color: var(--link); }
strong { font-weight: 650; }
.abstract { background: var(--panel); border: 1px solid var(--rule); border-left: 3px solid var(--accent); padding: 18px 22px; margin: 6px 0 10px; font-size: 16px; }
.abstract h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-3); padding: 0; }
.abstract p { max-width: none; margin: 0; }
code, pre { font-family: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace; font-size: 13.5px; }
code { background: var(--code); padding: 1px 5px; border-radius: 3px; }
pre { background: var(--code); border: 1px solid var(--rule); padding: 14px 16px; overflow-x: auto; border-radius: 4px; line-height: 1.5; }
pre code { background: none; padding: 0; }
.tablewrap { overflow-x: auto; margin: 10px 0 20px; border: 1px solid var(--rule); border-radius: 4px; }
table { border-collapse: collapse; font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 13.5px; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; background: var(--panel); position: sticky; top: 0; }
tr:last-child td { border-bottom: none; }
td:first-child, th:first-child { position: sticky; left: 0; background: var(--panel); }
figure.fig { margin: 22px 0 26px; padding: 0; }
figure.fig img { width: 100%; height: auto; display: block; border-radius: 4px; }
figcaption { font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 13px; color: var(--ink-2); margin-top: 8px; line-height: 1.45; max-width: 76ch; }
.pending { font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 13px; color: var(--ink-3); border: 1px dashed var(--rule-2); padding: 10px 14px; border-radius: 4px; margin: 14px 0; }
blockquote { margin: 0 0 14px; padding-left: 14px; border-left: 2px solid var(--rule-2); color: var(--ink-2); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 30px 0; }
.callout { background: var(--accent-soft); border-radius: 4px; padding: 12px 16px; margin: 14px 0; font-size: 15.5px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""


def build():
    src = (ROOT / "paper.md").read_text()
    # split off title / subtitle / meta (first three non-empty lines)
    lines = src.splitlines()
    title = lines[0].lstrip("# ").strip()
    subtitle = lines[2].strip().strip("*").strip()
    meta = lines[4].strip().strip("*").strip()
    body = "\n".join(lines[5:])

    # figure markers -> html
    for key in FIGURES:
        body = body.replace(f"<!-- {key} -->", figure_block(key))
    # leftover placeholders -> visible pending notes
    body = re.sub(r"<!-- ([A-Z0-9_]+) -->",
                  lambda m: f'<div class="pending">Section content "{m.group(1)}" is filled in once its experiment finishes.</div>', body)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "2-3", "anchorlink": False}})
    html = md.convert(body)
    # abstract block
    html = html.replace("<h2 id=\"abstract\">Abstract</h2>", "<section class=\"abstract\"><h2>Abstract</h2>", 1)
    html = html.replace("<h2 id=\"1-introduction\">", "</section><h2 id=\"1-introduction\">", 1)
    # wrap tables
    html = html.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    toc = md.toc

    page = f"""<title>Fractional Key Normalization</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<nav class="toc"><div class="eyebrow">Contents</div>{toc}</nav>
<main>
<header class="title">
<div class="kicker">daily-research-lab · technical report · 2026-09-01</div>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
<div class="meta">{meta}</div>
</header>
{html}
</main>
</div>
"""
    out = ROOT / "index.html"
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    build()
