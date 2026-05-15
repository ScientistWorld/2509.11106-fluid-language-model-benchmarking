# Problem Definition

## Research Question

Language model (LM) benchmarking faces several fundamental challenges: comprehensive evaluations are computationally expensive, benchmarks often fail to measure the capabilities they claim to measure, and evaluation quality degrades as benchmarks saturate when stronger models score near ceiling. Labeling errors in benchmark items are an additional motivation for better item-level evaluation, but this gym scores the four benchmark-quality dimensions below. The research question is: can we design evaluation methods that simultaneously improve efficiency, validity, variance, and saturation of LM benchmarks?

## Why It Matters

Current LM benchmarking often uses static evaluation sets: the same questions are asked of every model, regardless of whether all questions are equally informative for that model. This can be inefficient, noisy, and eventually ineffective as models improve and benchmarks stop differentiating them. Better evaluation methods would reduce computational cost, produce more reliable results, and maintain their usefulness as models improve.

## Success Criteria

A successful evaluation method should:
- **Improve efficiency**: achieve comparable or better quality estimates with fewer evaluation items
- **Increase validity**: produce scores that better predict performance on related capabilities (external validity)
- **Reduce variance**: produce more stable scores across repeated evaluations with the same model
- **Delay saturation**: maintain the ability to differentiate between models even as model capabilities improve

These properties matter because they determine how useful a benchmark is for tracking model progress, comparing models, and making decisions based on evaluation results.
