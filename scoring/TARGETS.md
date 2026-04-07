# Evaluation Targets

## Primary: saturation
FLUID should produce ability estimates that are well-correlated with ground-truth LLM performance, indicating the benchmark is not saturated.
- **Metric**: saturation (higher is better)
- **Evidence**: FLUID must outperform both RANDOM and RANDOM IRT baselines on saturation across all item counts (10, 50, 100)

## Constraint: validity
FLUID's ability estimates should correctly rank LLMs relative to each other (low rank distance from ground truth).
- **Metric**: rank_distance (lower is better)
- **Evidence**: FLUID must have lower rank distance than RANDOM baseline

## Constraint: variance
FLUID's ability estimates should be stable across repeated runs (low total variation).
- **Metric**: total_variation (lower is better)
- **Evidence**: FLUID must have lower total variation than both RANDOM and RANDOM IRT baselines

## Ablation: efficiency_scaling
FLUID's advantage over baselines should hold across different item budgets (10, 50, 100), demonstrating it works efficiently with few items.
- **Metric**: saturation (higher is better), at each item count
- **Evidence**: FLUID-10 should already beat RANDOM-100 on saturation; FLUID-50 and FLUID-100 should show diminishing returns
