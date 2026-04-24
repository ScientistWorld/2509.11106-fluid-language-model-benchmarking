# Experiment Overview

## table2_ablation_comparison
Tests whether a candidate efficient benchmarking strategy outperforms matched subset baselines on
the three reported evaluation metrics: validity, variance, and saturation. The comparison is run at
item budgets 10, 50, 100, and 500, averaged over 6 benchmarks, 6 language models, and all released
checkpoints for those models. This experiment is the minimum target for the core claim.

The result rows cover three strategy families at each item budget:
- A random subset baseline with standard accuracy aggregation.
- A random subset baseline with the benchmark's calibrated score aggregation.
- The candidate efficient benchmarking strategy at the same item budget.

## appendix_h_full_accuracy_comparison
Tests whether the candidate efficient strategy remains competitive with a full-benchmark accuracy
curve even when efficiency is not the only concern. The reproduced `full_accuracy` curve uses all
available item-level checkpoint responses with standard accuracy aggregation, while the candidate
row uses the 500-item run from the primary experiment.

The result rows compare full-benchmark static accuracy against the candidate strategy at a
500-item budget.

## appendix_f_table4_lm_breakdown_100
Tests whether the candidate strategy's gains persist across different LM families at the paper's
100-item setting. The reproduced rows are aggregated separately for each of the six released LMs,
averaging across the six benchmarks and all released checkpoints for that LM.

The result rows compare three strategy families per LM:
- A random subset baseline with standard accuracy aggregation.
- A random subset baseline with calibrated IRT aggregation.
- The candidate adaptive strategy at the same 100-item budget.
