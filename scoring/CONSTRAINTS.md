# Scientist Constraints

## Data
The scientist must use only the pre-fitted IRT model parameters (discrimination `a` and difficulty
`b` values) provided in the HuggingFace dataset `allenai/fluid-benchmarking`. These parameters
were fitted by the paper's authors using MCMC on item responses from 102 Open LLM Leaderboard
models. The scientist may not re-estimate IRT parameters on additional data or use IRT models
fitted on different corpora.

The LM checkpoint data (6 LMs × 61-94 checkpoints each × 6 benchmarks with item-level 0/1 responses)
must be used as-is. The scientist may not supplement with additional model checkpoints,
additional benchmarks, or human-labeled data beyond what is provided.

The set of evaluation benchmarks (ARC Challenge, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande)
and the 6 LM families (Amber-6.7B, K2-65B, OLMo1-7B, OLMo2-7B, Pythia-2.8B, Pythia-6.9B) define
the evaluation scope. The scientist may not expand to other benchmarks or models.

**Why**: The paper's IRT parameters are calibrated on the specific item responses from the Open LLM
Leaderboard. Re-fitting on different data or adding more models would change what the IRT model
represents, making comparisons to the paper's results meaningless.

## Item Budget
Any proposed improvement must be evaluated under the same item budgets as the paper: 10, 50, 100,
and 500 items. A method that only works well at 1000 items does not demonstrate the efficiency
contribution that FLUID provides.

**Why**: The paper's core claim is that FLUID outperforms baselines *at the same item count*. Beating
FLUID at 1000 items while losing at 100 items does not advance the paper's contribution.

## Evaluation Metrics
The scientist must use the paper's four evaluation dimensions: validity (rank distance), variance
(total variation), saturation (Spearman correlation), and efficiency (performance vs. item count).
The scientist may not redefine or rename these metrics to make a weaker method look better.

**Why**: These four dimensions capture the paper's full thesis. Optimizing only one (e.g., only
validity) while degrading others does not demonstrate genuine improvement.

## Compliance Checklist
- [ ] IRT model parameters used are from `allenai/fluid-benchmarking`, not re-fitted
- [ ] No external benchmarks, models, or datasets added beyond the provided data
- [ ] Results reported at item counts 10, 50, 100, and 500
- [ ] All four metrics (rank_distance, total_variation, saturation, efficiency) are reported
- [ ] Comparison includes RANDOM and RANDOM IRT baselines at matching item counts
