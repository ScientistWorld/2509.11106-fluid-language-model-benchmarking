# Scientist Constraints

## Data
The scientist must use only the reference-model item metadata and item-level responses provided in
the dataset bundled by this environment. The scientist may not estimate new item metadata on
additional data or use metadata fitted on different corpora.

The LM checkpoint data (6 LMs × 61-94 checkpoints each × 6 benchmarks with item-level 0/1 responses)
must be used as-is. The scientist may not supplement with additional model checkpoints,
additional benchmarks, or human-labeled data beyond what is provided.

The set of evaluation benchmarks (ARC Challenge, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande)
and the 6 LM families (Amber-6.7B, K2-65B, OLMo1-7B, OLMo2-7B, Pythia-2.8B, Pythia-6.9B) define
the evaluation scope. The scientist may not expand to other benchmarks or models.

**Why**: The reference information is calibrated on a specific benchmark-response corpus. Re-fitting
on different data or adding more models would change the target distribution and make comparisons
to the reproduced results meaningless.

## Item Budget
Any proposed improvement must be evaluated under the same item budgets as the paper: 10, 50, 100,
and 500 items. A method that only works well at 1000 items does not demonstrate the efficiency
claim being evaluated here.

**Why**: The core claim is improvement at the same item count. Beating the reproduced score at 1000
items while losing at 100 items does not address the target problem.

## Evaluation Metrics
The scientist must use the paper's four evaluation dimensions: validity (rank distance), variance
(total variation), saturation (Spearman correlation), and efficiency (performance vs. item count).
The scientist may not redefine or rename these metrics to make a weaker method look better.

**Why**: These four dimensions capture the paper's full thesis. Optimizing only one (e.g., only
validity) while degrading others does not demonstrate genuine improvement.

## Compliance Checklist
- [ ] Reference item metadata is from the provided dataset, not re-fitted
- [ ] No external benchmarks, models, or datasets added beyond the provided data
- [ ] Results reported at item counts 10, 50, 100, and 500
- [ ] All four metrics (rank_distance, total_variation, saturation, efficiency) are reported
- [ ] Comparison includes the two random subset baselines at matching item counts
