# Reproduction Checklist

Check off items as you complete them. Order doesn't matter — work however makes sense for this paper.

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers (Table 2 plus Appendix H)
- [x] Workspace validated via `python validate.py`
- [x] `scoring/scores.json` — reproduced numbers (Table 2 plus Appendix H)
- [x] `scoring/TARGETS.md` — primary, constraint, and ablation targets
- [x] `scoring/CONSTRAINTS.md` — what a scientist must hold fixed
- [x] `scoring/DIRECTION.md` — research direction scope for scientist
- [x] `scoring/EXPERIMENTS.md` — experiment descriptions

## Code
- [x] Evaluation code in `eval/` (compute_metrics, score_results, scoring, run_evaluation)
- [x] Method implementation in `method/` (irt, fluid, experiment, fluid_benchmarking)
- [x] Baseline implementations (integrated in method/fluid.py and wrapped by scripts/baseline.sh: RANDOM, RANDOM_IRT)

## Scripts
- [x] `scripts/evaluate.sh` — self-contained evaluation runner
- [x] `scripts/reproduce.sh` — end-to-end reproduction script
- [x] `scripts/download.sh` — idempotent download script with size comment
- [x] `scripts/method.sh` — runs FLUID method on experiment data
- [x] `scripts/baseline.sh` — runs matched RANDOM and RANDOM_IRT baselines as part of the shared ablation experiment
- [x] `scripts/run.sh` — job script for compute submission

## Environment
- [x] `environment/container.def` — Apptainer container definition (python:3.12-slim-bookworm)
- [x] `environment/setup.sh` — pip installs via uv (numpy, scipy, pandas)

## Data
- [x] Dataset acquired (from HuggingFace allenai/fluid-benchmarking)
- [x] Pretrained models downloaded if needed (paper releases pre-fit IRT parameters)

## Results
- [x] Core claim verified: FLUID beats RANDOM on all metrics (rank_distance, total_variation, saturation) across all 4 budgets (10, 50, 100, 500)
- [x] 12 Table 2 method-budget entries plus Appendix H full-accuracy comparison scored

## Milestones
- [x] `method_runs` — FLUID algorithm executes end-to-end without errors
- [x] `core_claim` — minimum experiment supports central claim (FLUID beats RANDOM across all metrics)
