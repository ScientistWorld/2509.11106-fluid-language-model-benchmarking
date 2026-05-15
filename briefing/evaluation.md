# Evaluation

## Metrics

Four dimensions of evaluation quality are measured:

1. **Efficiency**: Measured by varying the number of items used per benchmark (e.g., 10, 50, 100, 500 items). The goal is to achieve good quality with fewer items.

2. **Validity** (rank distance, lower is better): For pairs of benchmarks targeting similar capabilities (ARC Challenge to/from MMLU for knowledge/reasoning; HellaSwag to/from WinoGrande for commonsense reasoning), compute how well estimated performance on one benchmark predicts actual performance on the other. The comparison is directional and averaged both ways for each pair. GSM8K and TruthfulQA contribute to variance and saturation but are not used in the validity-pair calculation. Lower distance means better validity: the estimated scores are more predictive of performance on a related benchmark.

3. **Variance** (total variation, lower is better): Measures step-to-step stability of training curves. For one model/benchmark curve with T ordered checkpoints and scores x_t:
   - TV = (T/(T-1)) * sum_{t=1}^{T-1} |x_{t+1} - x_t| / |x_T - x_1|
   - If the denominator is effectively zero, the evaluator returns 0.0 for a flat curve and caps non-flat undefined curves at 100.0.
   Lower TV means lower variance and more reliable evaluation.

4. **Saturation** (Spearman rank correlation, higher is better): Measures monotonicity of training curves. Compute the absolute Spearman rank correlation between checkpoint sequence and predicted performance values. The score captures whether the benchmark still provides an ordered training-progress signal rather than a flat/noisy curve; in this setting, model checkpoints are expected to improve with training, so high monotonicity is interpreted as lower saturation.

## Comparison Targets

The primary comparison averages results across 6 benchmarks, 6 language models, and 61-94
checkpoints per model. The comparison includes fixed-size subset baselines and each submitted
method at the same item budgets. Future methods should produce the same output format so they can
be compared at equal item counts.

Target numbers for that comparison are:

| Method | Validity (↓) | Variance (↓) | Saturation (↑) |
|--------|-------------|-------------|----------------|
| RANDOM (10 items) | 20.0 | 29.0 | 0.47 |
| RANDOM (50 items) | 15.2 | 19.1 | 0.62 |
| RANDOM (100 items) | 16.9 | 19.8 | 0.64 |
| RANDOM (500 items) | 9.1 | 10.2 | 0.79 |
| Random reference-metadata baseline (10 items) | 14.1 | 18.2 | 0.48 |
| Random reference-metadata baseline (50 items) | 11.1 | 15.7 | 0.69 |
| Random reference-metadata baseline (100 items) | 10.6 | 17.8 | 0.71 |
| Random reference-metadata baseline (500 items) | 8.4 | 10.9 | 0.85 |
| Reproduced submitted method (10 items) | 10.1 | 10.7 | 0.76 |
| Reproduced submitted method (50 items) | 8.8 | 6.5 | 0.86 |
| Reproduced submitted method (100 items) | 8.7 | 6.1 | 0.85 |
| Reproduced submitted method (500 items) | 8.3 | 4.9 | 0.88 |

A strong submitted method should outperform both random baselines across all three reported metrics
and all four item budgets.

## Evaluation Protocol

1. **Reference phase**: Use only the provided reference-model benchmark results and calibration metadata
   supplied with this environment. Keep the six test-model families disjoint from this reference
   pool.

2. **Test models**: Evaluate 6 language models with publicly available checkpoints across
   training: Amber-6.7B (73 checkpoints), OLMo1-7B (83 checkpoints), OLMo2-7B (94 checkpoints),
   Pythia-6.9B (78 checkpoints), Pythia-2.8B (78 checkpoints), and K2-65B (61 checkpoints).

3. **Evaluation**: For each model, benchmark, and subset size (10, 50, 100, 500 items), evaluate
   all checkpoints under each compared strategy. Compute validity (rank distance), variance (total
   variation), and saturation (Spearman correlation).

4. **Aggregation**: Report averaged results across benchmarks and LMs for each method and subset size.
