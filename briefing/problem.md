# Problem Definition

## Research Question

Language model (LM) benchmarking faces several fundamental challenges: (1) comprehensive evaluations are computationally expensive, (2) benchmarks often fail to measure the capabilities they claim to measure, and (3) evaluation quality degrades due to labeling errors in benchmark items and benchmark saturation (when frontier models score near the maximum). The research question is: can we design evaluation methods that simultaneously improve efficiency, validity, variance, and saturation of LM benchmarks?

## Why It Matters

Current LM benchmarking uses static evaluation sets — the same questions are asked of every model, regardless of the model's capability level. This is inefficient (we evaluate small models with the same expensive set as large models), noisy (random item selection produces high variance), and eventually ineffective (as models improve, benchmarks saturate and stop differentiating). Better evaluation methods would reduce computational cost, produce more reliable results, and maintain their usefulness as models improve.

## Success Criteria

A successful evaluation method should:
- **Improve efficiency**: achieve comparable or better quality estimates with fewer evaluation items
- **Increase validity**: produce scores that better predict performance on related capabilities (external validity)
- **Reduce variance**: produce more stable scores across repeated evaluations with the same model
- **Delay saturation**: maintain the ability to differentiate between models even as model capabilities improve

These properties matter because they determine how useful a benchmark is for tracking model progress, comparing models, and making decisions based on evaluation results.
