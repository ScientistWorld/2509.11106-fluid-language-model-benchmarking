# Evaluation Targets

## Primary: validity
The candidate strategy should improve external validity relative to random subset evaluation.
- **Metric**: `rank_distance` (lower is better)
- **Experiment**: `table2_ablation_comparison`
- **Evidence**: `fluid_10`, `fluid_50`, `fluid_100`, and `fluid_500` should beat the corresponding `random_*` baselines, and typically also `random_irt_*`

## Constraint: variance
The candidate strategy should reduce step-to-step instability in checkpoint tracking.
- **Metric**: `total_variation` (lower is better)
- **Evidence**: `fluid_*` should beat both `random_*` and `random_irt_*` at the same item budget

## Constraint: saturation
The candidate strategy should delay benchmark saturation so late-training progress remains visible.
- **Metric**: `saturation` (higher is better)
- **Evidence**: `fluid_*` should have higher monotonicity than the random baselines, especially at small budgets

## Ablation: efficiency_scaling
The candidate strategy's advantage should persist from very small to moderate item budgets.
Efficiency is represented by the item-budget sweep (`10`, `50`, `100`, and `500` items), not by a
separate scalar metric.
- **Metric**: all three metrics at budgets `10`, `50`, `100`, and `500`
- **Evidence**: the directional advantage should remain visible as the item budget increases, with the largest gains at low budgets

## Secondary: full_benchmark_accuracy
The candidate strategy should remain competitive with, and preferably outperform, evaluating the
entire benchmark with standard accuracy aggregation.
- **Metric**: `rank_distance`, `total_variation`, and `saturation`
- **Experiment**: `appendix_h_full_accuracy_comparison`
- **Evidence**: `fluid_500` should beat `full_accuracy`, showing that a lower-budget strategy can
  improve evaluation quality even when full-benchmark accuracy uses more items
