# Problem Definition

## Research Question

Language model (LM) benchmarking faces several fundamental challenges: comprehensive evaluations are computationally expensive, benchmarks often fail to measure the capabilities they claim to measure, and evaluation quality degrades as benchmarks saturate when stronger models score near ceiling. Labeling errors in benchmark items are an additional motivation for better item-level evaluation, but this gym scores the four benchmark-quality dimensions below. The research question is: can we design evaluation methods that simultaneously improve efficiency, validity, variance, and saturation of LM benchmarks?

## Setting

Evaluating a model on a benchmark can be written as choosing which items to administer (selection) and turning the item-level results into one benchmark-level score (aggregation). The setting here is monitoring LM pretraining: each checkpoint of a training run is evaluated on each of six benchmarks (ARC Challenge, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande) with a small item budget (10, 50, 100 or 500 items instead of the full benchmark), and every checkpoint is evaluated on its own, as it would be during training.

Available to a method:
- the benchmarks' item lists, with item parameters (difficulty and discrimination) of a released 2-parameter logistic item response model fitted on 102 reference LMs from the Open LLM Leaderboard (the LMs studied here are not among them);
- the reference LMs' benchmark accuracies and fitted abilities;
- complete item-level responses of three pretraining runs (Amber-6.7B, OLMo1-7B, OLMo2-7B) for development.

At evaluation time a method only learns a checkpoint's answers to the items it chooses to query, within the budget. Three further pretraining runs (K2-65B, Pythia-2.8B, Pythia-6.9B) are held out for testing.

## Why It Matters

Current LM benchmarking often uses static evaluation sets: the same questions are asked of every model, regardless of whether all questions are equally informative for that model. This can be inefficient, noisy, and eventually ineffective as models improve and benchmarks stop differentiating them. Better evaluation methods would reduce computational cost, produce more reliable results, and maintain their usefulness as models improve.

## Success Criteria

A successful evaluation method should:
- **Improve efficiency**: achieve comparable or better quality estimates with fewer evaluation items
- **Increase validity**: produce scores that better predict performance on related capabilities (external validity)
- **Reduce variance**: produce more stable scores from checkpoint to checkpoint along a training run
- **Delay saturation**: keep showing monotone progress along a training run even as model capabilities improve

These properties matter because they determine how useful a benchmark is for tracking model progress, comparing models, and making decisions based on evaluation results.
