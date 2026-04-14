# Experiment Overview

## table2_ablation_comparison
Tests whether the proposed adaptive benchmarking method outperforms two matched ablations on the
paper's three reported evaluation metrics: validity, variance, and saturation. The comparison is
run at item budgets 10, 50, 100, and 500, averaged over 6 benchmarks, 6 language models, and all
released checkpoints for those models. This reproduces the paper's Table 2 ablation result, which
is the minimum experiment for the core claim; Table 1's broader AP/TB/MB/SM/MA baseline family is
documented in the paper but is not part of this minimum scoring target.

Reported methods:
- `random_*`: random subset evaluation with accuracy aggregation
- `random_irt_*`: random subset evaluation with latent ability aggregation
- `fluid_*`: adaptive item selection with latent ability aggregation
