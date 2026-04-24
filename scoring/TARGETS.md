# Evaluation Targets

## Primary: validity
The candidate strategy should improve external validity relative to random subset evaluation.
- **Metric**: `rank_distance` (lower is better)
- **Experiment**: `table2_ablation_comparison`
- **Evidence**: the candidate rows at 10, 50, 100, and 500 items should beat the corresponding random subset baselines, and typically also the calibrated random subset baselines

## Constraint: variance
The candidate strategy should reduce step-to-step instability in checkpoint tracking.
- **Metric**: `total_variation` (lower is better)
- **Evidence**: the candidate rows should beat both matched baselines at the same item budget

## Constraint: saturation
The candidate strategy should delay benchmark saturation so late-training progress remains visible.
- **Metric**: `saturation` (higher is better)
- **Evidence**: the candidate rows should have higher monotonicity than the random baselines, especially at small budgets

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
- **Evidence**: the 500-item candidate row should beat the full-accuracy baseline, showing that a lower-budget strategy can
  improve evaluation quality even when full-benchmark accuracy uses more items

## Secondary: per_lm_generalization
The candidate strategy's advantage should persist across different LM families, not just in the
global average.
- **Metric**: `rank_distance`, `total_variation`, and `saturation`
- **Experiment**: `appendix_f_table4_lm_breakdown_100`
- **Evidence**: at the 100-item budget, the candidate rows should beat the matched random baselines
  for most individual LMs, especially on total variation and saturation
