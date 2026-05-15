# Experiment Overview

## table2_ablation_comparison
Tests whether a submitted fixed-budget benchmarking method outperforms matched subset baselines on
the three reported evaluation metrics: validity, variance, and saturation. The comparison is run at
item budgets 10, 50, 100, and 500, averaged over 6 benchmarks, 6 language models, and all released
checkpoints for those models. This experiment is the minimum target for the core claim.

The result rows cover three strategy families at each item budget:
- A random subset baseline with standard accuracy aggregation.
- A random subset baseline scored with the provided reference item metadata.
- The submitted fixed-budget method at the same item budget.

## appendix_h_full_accuracy_comparison
Tests whether the submitted fixed-budget method remains competitive with a full-benchmark accuracy
curve even when efficiency is not the only concern. The reproduced `full_accuracy` curve uses all
available item-level checkpoint responses with standard accuracy aggregation, while the submitted
method row uses the 500-item run from the primary experiment.

The result rows compare full-benchmark static accuracy against the submitted method at a
500-item budget.
