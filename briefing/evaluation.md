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

## Baselines and Targets

The paper compares against multiple prior benchmark refinement methods:
- **RANDOM**: Random item sampling (common practice for efficient evaluation)
- **RANDOM IRT**: IRT-based ability estimation on random items (ablates dynamic selection)
- **ANCHOR POINTS**: Item clustering-based selection
- **TINYBENCHMARKS**: IRT-based subset selection
- **METABENCH**: Information filtering-based selection
- **SMART**: Hard version selection (for ARC Challenge)
- **MAGI**: Hard version selection (for MMLU)
- **FLUID**: The proposed method combining IRT ability estimation with dynamic item selection

Key target numbers (from paper's Table 2, averaged across 6 benchmarks, 6 LMs, 61-94 checkpoints per LM):

| Method | Validity (↓) | Variance (↓) | Saturation (↑) |
|--------|-------------|-------------|----------------|
| RANDOM (10 items) | 20.0 | 29.0 | 0.47 |
| RANDOM (50 items) | 15.2 | 19.1 | 0.62 |
| RANDOM (100 items) | 16.9 | 19.8 | 0.64 |
| RANDOM IRT (10 items) | 14.1 | 18.2 | 0.48 |
| RANDOM IRT (50 items) | 11.1 | 15.7 | 0.69 |
| RANDOM IRT (100 items) | 10.6 | 17.8 | 0.71 |
| RANDOM IRT (500 items) | 8.4 | 10.9 | 0.85 |
| FLUID (10 items) | 10.1 | 10.7 | 0.76 |
| FLUID (50 items) | 8.8 | 6.5 | 0.86 |
| FLUID (100 items) | 8.7 | 6.1 | 0.85 |
| FLUID (500 items) | 8.3 | 4.9 | 0.88 |

FLUID outperforms RANDOM and RANDOM IRT across all three reported metrics and all four item budgets.

## Evaluation Protocol

1. **IRT Model Training**: Fit a 2-parameter logistic (2PL) item response model to existing LM evaluation results from the Open LLM Leaderboard (102 pretrained LMs, excluding the test models and related family members). Fit separate unidimensional IRT models per benchmark (ARC Challenge, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande).

2. **Test LMs**: Evaluate 6 LMs with publicly available checkpoints across training: Amber-6.7B (73 checkpoints), OLMo1-7B (83 checkpoints), OLMo2-7B (94 checkpoints), Pythia-6.9B (78 checkpoints), Pythia-2.8B (78 checkpoints), K2-65B (61 checkpoints).

3. **Evaluation**: For each LM, benchmark, and subset size (10, 50, 100, 500 items), evaluate all checkpoints using each method. Compute validity (rank distance), variance (total variation), and saturation (Spearman correlation).

4. **Aggregation**: Report averaged results across benchmarks and LMs for each method and subset size.
