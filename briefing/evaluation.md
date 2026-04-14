# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocol for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

Four dimensions of evaluation quality are measured:

1. **Efficiency**: Measured by varying the number of items used per benchmark (e.g., 10, 50, 100, 500 items). The goal is to achieve good quality with fewer items.

2. **Validity** (rank distance, lower is better): For pairs of benchmarks targeting similar capabilities (ARC Challenge ↔ MMLU for knowledge/reasoning; HellaSwag ↔ WinoGrande for commonsense reasoning), compute how well estimated performance on one benchmark predicts actual performance on the other. Specifically, compute the absolute rank distance between predicted rankings (based on estimated scores) and actual rankings (based on accuracy on the second benchmark). Lower distance means better validity — the estimated scores are more predictive of true performance.

3. **Variance** (total variation, lower is better): Measures step-to-step stability of training curves. For a model evaluated at multiple checkpoints on a benchmark, compute the normalized total variation:
   - Let x_i^t be the measured performance of model m_i at checkpoint t on benchmark Q
   - TV = (n/(n-1)) × Σ|x_i^{t+1} - x_i^t| / |x_i^n - x_i^1|
   Lower TV means lower variance and more reliable evaluation.

4. **Saturation** (Spearman rank correlation, higher is better): Measures monotonicity of training curves. Compute the absolute Spearman rank correlation between checkpoint sequence and predicted performance values. Higher correlation means the benchmark consistently reflects improvement through training — it has not yet saturated.

## Comparison Targets

One reported comparison in the paper averages results across 6 benchmarks, 6 language models, and
61-94 checkpoints per model. The comparison includes fixed-size subset baselines and a candidate
evaluation strategy at the same item budgets. Future methods should produce the same output format
so they can be compared at equal item counts.

Target numbers for that comparison are:

| Method | Validity (↓) | Variance (↓) | Saturation (↑) |
|--------|-------------|-------------|----------------|
| RANDOM (10 items) | 20.0 | 29.0 | 0.47 |
| RANDOM (50 items) | 15.2 | 19.1 | 0.62 |
| RANDOM (100 items) | 16.9 | 19.8 | 0.64 |
| RANDOM score baseline (10 items) | 14.1 | 18.2 | 0.48 |
| RANDOM score baseline (50 items) | 11.1 | 15.7 | 0.69 |
| RANDOM score baseline (100 items) | 10.6 | 17.8 | 0.71 |
| RANDOM score baseline (500 items) | 8.4 | 10.9 | 0.85 |
| Candidate (10 items) | 10.1 | 10.7 | 0.76 |
| Candidate (50 items) | 8.8 | 6.5 | 0.86 |
| Candidate (100 items) | 8.7 | 6.1 | 0.85 |
| Candidate (500 items) | 8.3 | 4.9 | 0.88 |

The candidate strategy should outperform both random baselines across all three reported metrics
and all four item budgets.

## Evaluation Protocol

1. **Reference phase**: Use only the provided reference-model benchmark results and item metadata
   supplied with this environment. Keep the six test-model families disjoint from this reference
   pool.

2. **Test models**: Evaluate 6 language models with publicly available checkpoints across
   training: Amber-6.7B (73 checkpoints), OLMo1-7B (83 checkpoints), OLMo2-7B (94 checkpoints),
   Pythia-6.9B (78 checkpoints), Pythia-2.8B (78 checkpoints), and K2-65B (61 checkpoints).

3. **Evaluation**: For each model, benchmark, and subset size (10, 50, 100, 500 items), evaluate
   all checkpoints under each compared strategy. Compute validity (rank distance), variance (total
   variation), and saturation (Spearman correlation).

4. **Aggregation**: Report averaged results across benchmarks and LMs for each method and subset size.
