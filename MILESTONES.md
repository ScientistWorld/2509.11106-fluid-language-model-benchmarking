# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-08 19:05] - none
- Audited the paper and current workspace against the released `allenai/fluid-benchmarking` codebase
- Confirmed the released artifact dataset is already downloaded under `data/fluid-benchmarking`
- Identified invalid hardcoded scoring in `scripts/evaluate.sh` and method/evaluation mismatches that needed correction before any milestone claim

### [2026-04-08 21:59] - method_runs
- Ran the paper's actual FLUID algorithm end to end on the released benchmark artifacts using a smoke-test subset of checkpoints
- Verified the pipeline writes `scoring/scores.json` in the required experiment-based schema
- Prepared the entrypoint scripts and local dependency path for a full released-data CPU reproduction run

## Stop Justification
