# Design Notes - FLUID BENCHMARKING Train/Test Evaluation

## What I Checked

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`: train calls `eval/train/score_slice.py` and writes `scoring/scores_train.json`; test calls `eval/test/score_slice.py` and writes `scoring/scores_test.json`.
- Checked for hardcoded method defaults: both slice scorers enumerate methods from the result payload and do not use a fixed `METHOD_NAME`.
- Checked train/test leakage with grep: no executable cross-imports between `eval/train/` and `eval/test/`; only docstring mentions remain.
- Checked `scoring/reference.json` metric entries: active metrics have `higher_is_better`, `coefficient`, and `description`, and none are paper-agreement/error-to-publication metrics.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist at handoff.

## What Changed

- Fixed a scientific-fidelity issue in both slice scorers: `rank_distance` now uses the paper-style OLL reference-rank calculation instead of the previous checkpoint-only ordinal-rank proxy.
- Strengthened the hidden test slice against metric-gaming: train validity still uses the visible paper pairs `arc_challenge`/`mmlu` and `hellaswag`/`winogrande`, while test validity uses held-out pairs `gsm8k`/`mmlu` and `arc_challenge`/`truthfulqa_mc2`.
- Applied the OLL-reference rank-distance calculation consistently to Table 2, Appendix H full-accuracy comparison, and Appendix F/Table 4 per-LM breakdown.
- Updated `scoring/reference.json` descriptions for `rank_distance` so the schema matches the actual OLL-reference rank metric.

## Runtime Concerns For Next Iteration

- I did not run the evaluators or generate scores in this design pass. The runner should execute both scripts and check that the new held-out benchmark-pair split remains calibrated rather than overly punitive.
- The test validity pairs intentionally differ from train, so train/test `rank_distance` values will no longer be directly pair-identical. The expected signal is stable method ordering with some absolute-value shift.
- The slice scorers now require `data/downloads/fluid-benchmarking/open_llm_leaderboard_results.json`. The file is present in this workspace; the runner should confirm packaged environments always obtain it through `scripts/download.sh`.
- Table 4 still uses all LM result rows in both slices but splits checkpoints hash-wise. This preserves matching result identifiers, but subgroup-overfit protection for that experiment mainly comes from held-out checkpoints and held-out benchmark pairs, not held-out LM names.

## Static Checks Completed

- `bash -n scripts/evaluate_train.sh`
- `bash -n scripts/evaluate_test.sh`
- grep checks for train/test cross-imports, hardcoded method names, paper-agreement metric names, and stale train/test score stubs
