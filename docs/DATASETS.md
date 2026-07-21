# Tiny dataset catalog (the fuel)

Everything here loads and shows signal in minutes on a CPU. Generator tasks need no download (infinite, tunable). Grouped by type; see the daily rotation at the bottom.

## Generated / synthetic (no download)
- **Copy / Reverse / Sort** — cleanest architecture separator. `np.random.randint` + reorder. sec–min.
- **n-digit Addition** — carry algorithm + length-gen. minGPT `projects/adder`. minutes.
- **Modular arithmetic (grokking)** — `(a∘b) mod p`; p=97 → 9,409 pairs, train on ~30–50%. Delayed generalization; can take 10⁴–10⁵ steps (leave running).
- **Parity / counting** — XOR over bits; easy for RNNs, hard for vanilla attention. seconds.
- **Associative recall / MQAR** — the attention-vs-SSM litmus test. [zoology](https://github.com/HazyResearch/zoology). minutes.
- **Needle-in-a-haystack** — retrieve a planted fact from distractors. [RULER](https://github.com/NVIDIA/RULER) generator. minutes.
- **ListOps-mini** — nested MAX/MIN/SUM; hierarchical reasoning. minutes.
- **Cellular automata (Game of Life)** — learn/iterate a rule; depth generalization. <10k params. minutes.
- **Dyck brackets** — stack depth / matching; great for probing. minutes.
- **ProntoQA / ProsQA** — multi-step logical deduction; every step checkable. [prontoqa](https://github.com/asaparov/prontoqa), [coconut](https://github.com/facebookresearch/coconut). minutes.

## Tiny text
- **tiny-shakespeare** — ~1.1 MB, 65-char vocab. `karpathy/tiny_shakespeare`. min–30 min.
- **enwik8 slice** — first 1–5 MB of 100 MB. http://mattmahoney.net/dc/enwik8.zip. minutes/epoch.
- **TinyStories (subset)** — full is 7.6 GB, so stream a few thousand. `roneneldan/TinyStories` (CDLA-Sharing-1.0). tens of min.

## Small puzzles
- **4×4 Sudoku** — only 288 valid grids; generate + mask. minutes. (9×9 is CPU-marginal.)
- **Small mazes** — DFS/Prim generate; BFS labels shortest path. minutes.
- **ARC-AGI sample** — tiny JSON; 400+400 (v1), 1000+120 (v2). Apache-2.0. Probe/search, not train-to-solve.
- **Countdown** — reach target via +−×÷. [TinyZero](https://github.com/Jiayi-Pan/TinyZero) generator. minutes.

## Tabular / symbolic
- **Feynman equations (SRSD/SRBench)** — recover physics formulas. [srsd](https://github.com/omron-sinicx/srsd-benchmark). sec–min.
- **UCI via sklearn** — Iris/Wine/BreastCancer/Digits, bundled, KB-scale. sub-second.

## Small vision
- **MNIST** — 70k×28×28, ~11 MB. torchvision. <1 min/epoch MLP.
- **Fashion-MNIST** — harder drop-in (MIT). minutes.
- **CIFAR-10 subset** — subset 5–10k for CPU. minutes.
- **dSprites** — 64×64 binary, labeled factors, ~26 MB (Apache-2.0). minutes.

## Toy RL (CPU-native)
- **CartPole / Acrobot / Pendulum** — `gymnasium` classic control. solves in sec–min.
- **FrozenLake / gridworlds** — tabular Q-learning. seconds.
- **MiniGrid** — partial-obs navigation (Apache-2.0). small envs in minutes.

## Reasoning / QA
- **bAbI** — 20 unit-test reasoning tasks, few MB (BSD). minutes.
- **GSM8K** — 7,473+1,319, ~5.9 MB (MIT). Eval/format harness, not train-to-solve on CPU.

## Daily rotation (cycle one per day)
1 Copy/Reverse/Sort · 2 n-digit Addition · 3 MQAR recall · 4 tiny-shakespeare · 5 MNIST/Fashion · 6 dSprites/shapes · 7 CartPole · 8 bAbI · 9 ProntoQA/Countdown · 10 Feynman/UCI.
Big-swing overnight slots: grokking, ARC-AGI search, 9×9 Sudoku.
