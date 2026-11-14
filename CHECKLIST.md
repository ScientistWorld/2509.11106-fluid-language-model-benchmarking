# Reproduction Checklist

Check off items as you complete them. Order doesn't matter — work however makes sense for this paper.

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers (Table 2, Appendix H, plus Appendix F reference coverage)
- [x] Workspace validated via `python validate.py`
- [x] `scoring/scores.json` / `scores_train.json` / `scores_test.json` — reference method scored by the evaluators (visible LMs / held-out LMs)
- [x] `scoring/TARGETS.md` — primary, constraint, and ablation targets
- [x] `scoring/CONSTRAINTS.md` — what a scientist must hold fixed
- [x] `scoring/DIRECTION.md` — research direction scope for scientist
- [x] `scoring/EXPERIMENTS.md` — experiment descriptions

## Code
- [x] Evaluation code in `eval/train/score.py` and `eval/test/score.py` (independent; run the method through the item-query protocol)
- [x] Method implementation in `method/` (irt, fluid, serve = protocol server)
- [x] Baselines: computed by the evaluators (RANDOM, RANDOM IRT, full accuracy, all-item ability); `scripts/baseline.sh` serves the ablations through the protocol

## Scripts
- [x] `scripts/evaluate.sh` / `evaluate_train.sh` / `evaluate_test.sh` — no-argument evaluators
- [x] `scripts/reproduce.sh` — end-to-end reproduction script
- [x] `scripts/download.sh` — idempotent download script with size comment
- [x] `scripts/method.sh` — the submission (FLUID protocol server)
- [x] `scripts/baseline.sh` — RANDOM / RANDOM IRT served through the protocol
- [x] `scripts/run.sh` — job script for compute submission

## Environment
- [x] `environment/container.def` — Apptainer container definition (ubuntu:22.04 with only OS-level packages)
- [x] `environment/setup.sh` — installs Python runtime dependencies with `uv pip install --system`

## Data
- [x] Dataset acquired (from HuggingFace allenai/fluid-benchmarking)
- [x] Pretrained models downloaded if needed (paper releases pre-fit IRT parameters)

## Results
- [x] Core claim verified: FLUID beats RANDOM on all metrics (rank_distance, total_variation, saturation) across all 4 budgets (10, 50, 100, 500)
- [x] Table 2 (4 budgets), Appendix H and worst-LM Table 4 experiments scored; hardening audit in `HARDENING.md`
- [x] Packaged container path validated on cluster (jobs c6167f00-2ce, 69bcfe1b-31d)

## Milestones
- [x] `method_runs` — FLUID algorithm executes end-to-end without errors
- [x] `core_claim` — minimum experiment supports central claim (FLUID beats RANDOM across all metrics)
- [x] `secondary_claims` — packaged run also reproduces the Appendix H full-accuracy comparison
