# Evaluation Targets

## Primary: validity
The submitted method should improve external validity relative to random subset evaluation.
- **Metric**: `rank_distance` (lower is better)
- **Experiment**: `table2_ablation_comparison`
- **Evidence**: the submitted-method rows at 10, 50, 100, and 500 items should beat the corresponding random subset baselines, and typically also the reference-metadata random subset baselines

## Constraint: variance
The submitted method should reduce step-to-step instability in checkpoint tracking.
- **Metric**: `total_variation` (lower is better)
- **Evidence**: the submitted-method rows should beat both matched baselines at the same item budget

## Constraint: saturation
The submitted method should delay benchmark saturation so late-training progress remains visible.
- **Metric**: `saturation` (higher is better)
- **Evidence**: the submitted-method rows should have higher monotonicity than the random baselines, especially at small budgets

## Ablation: efficiency_scaling
The submitted method's advantage should persist from very small to moderate item budgets.
Efficiency is represented by the item-budget sweep (`10`, `50`, `100`, and `500` items), not by a
separate scalar metric.
- **Metric**: all three metrics at budgets `10`, `50`, `100`, and `500`
- **Evidence**: the directional advantage should remain visible as the item budget increases, with the largest gains at low budgets

## Secondary: full_benchmark_accuracy
The submitted method should outperform evaluating the entire benchmark with standard accuracy
aggregation on the scored quality metrics.
- **Metric**: `rank_distance`, `total_variation`, and `saturation`
- **Experiment**: `appendix_h_full_accuracy_comparison`
- **Evidence**: the 500-item submitted-method row should beat the full-accuracy baseline, showing that a lower-budget strategy can
  improve evaluation quality even when full-benchmark accuracy uses more items
