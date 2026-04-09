# Experiment Overview

## table2_ablation_comparison
Tests whether the proposed adaptive benchmarking method outperforms two matched ablations on the
paper's three reported evaluation metrics: validity, variance, and saturation. The comparison is
run at item budgets 10, 50, 100, and 500, averaged over 6 benchmarks, 6 language models, and all
released checkpoints for those models.

Reported methods:
- `random_*`: random subset evaluation with accuracy aggregation
- `random_irt_*`: random subset evaluation with latent ability aggregation
- `fluid_*`: adaptive item selection with latent ability aggregation
