# Progress

## What Works

- Paper analysis complete: FLUID BENCHMARKING combines 2PL IRT with adaptive item selection using Fisher information.
- Briefing files complete: problem.md, evaluation.md, method.md, overview.md.
- Method implementation in `method/`: 2PL IRT (irt.py), adaptive selection (fluid.py), metrics (metrics.py), experiment runner (experiment.py).
- Evaluation code in `eval/`: compute_metrics.py, score_results.py, scoring.py.
- Data loaded from HuggingFace allenai/fluid-benchmarking: pre-fit IRT models, LM checkpoint results (6 LMs × 61-94 checkpoints × 6 benchmarks), OLL leaderboard results.
- Container definition (container.def) and setup (setup.sh) ready.
- All scripts (run.sh, reproduce.sh, evaluate.sh, method.sh, download.sh) in place.
- Full experiment submission: all 6 LMs, all checkpoints (61-94 each), 4 budgets (10/50/100/500), 3 methods (RANDOM, RANDOM IRT, FLUID).

## Results

- `method_runs` confirmed via smoke test on login node (80s for 3 checkpoints × 2 budgets).
- Full reproduction job submitted: estimated ~1 hour runtime.

## Deviations from Paper

- Reproduction uses the paper's released pre-fit 2PL item parameters and released checkpoint/item-response artifacts rather than re-fitting the 2PL models from raw Open LLM Leaderboard item-level responses. This preserves the paper's evaluation algorithm while reducing compute and data-assembly burden.

## Remaining

- Await full experiment results.
- Compare reproduced metrics against paper's Table 2 numbers.
- Target: `core_claim` — FLUID beats RANDOM on validity (rank_distance) and variance (total_variation).

## Issues

- Login node has no system numpy; relies on `/home/user/.deps/fluid_benchmarking` overlay.
- Full experiment submitted as CPU job.
