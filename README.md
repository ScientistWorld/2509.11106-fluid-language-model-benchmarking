# Fluid Language Model Benchmarking

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening).

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.11106"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.11106-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2509.11106](https://arxiv.org/abs/2509.11106) |
|:---|:---|
| **Domain** | NLP / LM Evaluation Methodology |
| **TL;DR** | A new LM benchmarking method that adapts evaluation items to each model's capability level, improving efficiency, validity, variance, and saturation simultaneously. |

## Summary

This paper identifies four fundamental problems with current LM benchmarking: it's computationally expensive, often fails to measure what it claims (low validity), produces noisy results (high variance), and becomes useless as models improve (saturation). The paper proposes FLUID BENCHMARKING, which combines item response theory (IRT) with adaptive testing techniques from education. The key idea is that the best questions to ask depend on the model's capability — easy questions for weak models, hard questions for strong models. FLUID fits a psychometric model to existing evaluation data, then dynamically selects the most informative items for each model being evaluated. This produces better ability estimates using far fewer items.

## Problem

Language model (LM) benchmarking faces several fundamental challenges: comprehensive evaluations are computationally expensive, benchmarks often fail to measure the capabilities they claim to measure, and evaluation quality degrades as benchmarks saturate when stronger models score near ceiling. Labeling errors in benchmark items are an additional motivation for better item-level evaluation, but this gym scores the four benchmark-quality dimensions below. The research question is: can we design evaluation methods that simultaneously improve efficiency, validity, variance, and saturation of LM benchmarks?

## The paper's method

FLUID BENCHMARKING is a new LM evaluation method that improves benchmarking across four dimensions (efficiency, validity, variance, saturation) by combining two ideas from psychometrics: (1) item response theory (IRT) for measuring ability in a latent space, and (2) computerized adaptive testing for dynamic item selection.

## Key results of the paper

- FLUID BENCHMARKING substantially improves over RANDOM at the same item budget; for example, at 10 items its mean validity is 10.1 versus 20.0 and its mean variance is 10.7 versus 29.0
- FLUID reduces variance by 3-5x compared to random sampling across all subset sizes
- FLUID maintains high saturation (Spearman correlation 0.76-0.97) even as models improve, while random sampling saturates early
- IRT aggregation accounts for validity gains; dynamic item selection accounts for variance reductions

## Run it

```bash
pip install sciworld
sw research 2509.11106 --mode innovate --agent claude-code
sw evaluate 2509.11106
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

