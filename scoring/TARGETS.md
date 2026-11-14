# Evaluation Targets

Every target is measured on the scored rows of `scoring/scores_test.json` (hidden LMs) and, for
development, `scoring/scores_train.json` (visible LMs). Comparisons against baselines use the
evaluator's own `baselines` in the same file.

## Primary: validity
The submitted method's checkpoint scores should rank checkpoints consistently with a related
benchmark.
- **Metric**: `rank_distance` (lower is better)
- **Experiment**: `table2_ablation_comparison`, rows `submitted_10/50/100/500`
- **Evidence**: lower than `random_<budget>` and `random_irt_<budget>` at every budget

## Constraint: item budget (efficiency)
At most 10, 50, 100 or 500 item queries per checkpoint-benchmark session.
- **Enforced by**: the evaluator's query counter; one query beyond the budget fails the run and
  every scored metric is written at its worst value
- **Evidence**: `evaluation.mean_items_queried` in the score file

## Constraint: variance
The submitted method should keep step-to-step instability of checkpoint curves low.
- **Metric**: `total_variation` (lower is better); constant or endpoint-equal curves score 100
- **Experiment**: `table2_ablation_comparison`
- **Evidence**: lower than both matched baselines at every budget

## Constraint: saturation
Late-training progress should remain visible.
- **Metric**: `saturation` (higher is better; a constant curve scores 0)
- **Experiment**: `table2_ablation_comparison`
- **Evidence**: higher than both matched baselines at every budget

## Ablation: efficiency_scaling
The advantage should persist from 10 to 500 items, with the largest gains at small budgets.
- **Metric**: all three metrics across `submitted_10` ... `submitted_500` against the baselines

## Secondary: full_benchmark_accuracy
A 500-item evaluation should beat full-benchmark accuracy on all three metrics.
- **Experiment**: `appendix_h_full_accuracy_comparison`, row `submitted_500` against `baselines.full_accuracy`

## Secondary: per_model_consistency
The quality should hold for every training trajectory, not only on average.
- **Experiment**: `appendix_f_table4_worst_lm_100`, row `submitted_100` (worst LM per metric);
  `per_lm` shows each LM against both baselines
