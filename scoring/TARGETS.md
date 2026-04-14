# Evaluation Targets

## Primary: validity
FLUID should improve external validity relative to random subset evaluation.
- **Metric**: `rank_distance` (lower is better)
- **Experiment**: `table2_ablation_comparison`
- **Evidence**: `fluid_10`, `fluid_50`, `fluid_100`, and `fluid_500` should beat the corresponding `random_*` baselines, and typically also `random_irt_*`

## Constraint: variance
FLUID should reduce step-to-step instability in checkpoint tracking.
- **Metric**: `total_variation` (lower is better)
- **Evidence**: FLUID should beat both `random_*` and `random_irt_*` at the same item budget

## Constraint: saturation
FLUID should delay benchmark saturation so late-training progress remains visible.
- **Metric**: `saturation` (higher is better)
- **Evidence**: FLUID should have higher monotonicity than the random baselines, especially at small budgets

## Ablation: efficiency_scaling
FLUID's advantage should persist from very small to moderate item budgets. Efficiency is represented
by the item-budget sweep (`10`, `50`, `100`, and `500` items), not by a separate scalar metric.
- **Metric**: all three metrics at budgets `10`, `50`, `100`, and `500`
- **Evidence**: the directional advantage should remain visible as the item budget increases, with the largest gains at low budgets
