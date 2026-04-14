# Experiment Overview

## table2_ablation_comparison
Tests whether a candidate efficient benchmarking strategy outperforms matched subset baselines on
the three reported evaluation metrics: validity, variance, and saturation. The comparison is run at
item budgets 10, 50, 100, and 500, averaged over 6 benchmarks, 6 language models, and all released
checkpoints for those models. This experiment is the minimum target for the core claim.

Reported methods:
- `random_*`: random subset evaluation with accuracy aggregation
- `random_irt_*`: random subset evaluation with the reference score format used by the benchmark
- `fluid_*`: candidate strategy at the same item budget

## appendix_h_full_accuracy_comparison
Tests whether the candidate efficient strategy remains competitive with a full-benchmark accuracy
curve even when efficiency is not the only concern. The reproduced `full_accuracy` curve uses all
available item-level checkpoint responses with standard accuracy aggregation, while `fluid_500`
uses the 500-item candidate-strategy run from the primary experiment.

Reported methods:
- `full_accuracy`: full-benchmark static evaluation with accuracy aggregation
- `fluid_500`: candidate strategy at a 500-item budget
