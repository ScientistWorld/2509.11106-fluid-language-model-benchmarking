# Design Notes — FLUID BENCHMARKING Train/Test Evaluation

## Final Split

The shipped split uses two different mechanisms:
- `table2_ablation_comparison` and `appendix_h_full_accuracy_comparison` split by LM family.
- `appendix_f_table4_lm_breakdown_100` keeps all LM rows in both slices but splits each LM's checkpoints into disjoint even/odd subsets.

Train slice:
- LMs: `amber-7b`, `olmo1-7b`, `olmo2-7b`
- Table 4 checkpoints: even-indexed checkpoints within each LM curve

Test slice:
- LMs: `k2-65b`, `pythia-3b`, `pythia-7b`
- Table 4 checkpoints: odd-indexed checkpoints within each LM curve

This preserves the anti-shortcut goal: train is visible and optimized against, while test probes unseen LM families for the aggregate experiments and unseen checkpoints for the per-LM breakdown.

## Real-Run Results

The evaluation job completed successfully and wrote both `scoring/scores_train.json` and `scoring/scores_test.json`.

Coverage:
- `table2_ablation_comparison`: 12 train rows, 12 test rows
- `appendix_h_full_accuracy_comparison`: 2 train rows, 2 test rows
- `appendix_f_table4_lm_breakdown_100`: 18 train rows, 18 test rows
- Result identifier sets match exactly across train and test for every experiment

Signal quality on the frozen method:
- Table 2: `fluid` beats both matched baselines on all 24 metric comparisons in train and all 24 in test
- Appendix H: `fluid_500` beats `full_accuracy` on `total_variation` and `saturation` in both slices, and on `rank_distance` only in test
- Table 4: `fluid` beats matched baselines on 36/36 train metric comparisons and 34/36 test comparisons

## Checks Confirmed

### Scripts
- `scripts/evaluate_train.sh` writes `scores_train.json`
- `scripts/evaluate_test.sh` writes `scores_test.json`
- `scripts/run.sh` scores both slices from the frozen `checkpoints/experiment_results.json` payload
- No method name is hardcoded; scorers enumerate available methods from the payload

### Leakage
- `eval/train/score_slice.py` and `eval/test/score_slice.py` remain self-contained
- No cross-imports between train and test
- No metric is defined as agreement with the paper

### Calibration
- The split is not byte-identical: average absolute train-test metric gap is `2.861` for Table 2, `2.736` for Appendix H, and `1.809` for Table 4
- The method ranking signal remains stable across slices instead of collapsing

## Main Caveat

The largest train-test deltas are in `total_variation`, especially:
- `random_irt_50` in Table 2: `16.11` train vs `41.32` test
- Amber rows in Table 4: test `total_variation` is much harsher than train
- `full_accuracy` in Appendix H: `4.97` train vs `17.13` test

These do not break the benchmark, but they mean test is materially more punitive on volatility than train for some rows. That caveat should remain documented for downstream users.
