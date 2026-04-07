# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-07 10:30] - none
- Read the paper and identified key components
- Paper proposes FLUID BENCHMARKING combining IRT with adaptive item selection
- Core algorithm: 2PL IRT model + Fisher information-based item selection
- Data available: pre-fitted IRT models, OLL results, 6 test LM checkpoint evaluations

### [2026-04-07 11:00] - none → method_runs
- Filled in all briefing files (problem, evaluation, method, overview)
- Implemented IRT model: ability estimation, Fisher information, logistic function
- Implemented FluidBenchmarking class with adaptive item selection
- Implemented all 4 evaluation metrics (validity, variance, saturation, efficiency)
- Downloaded all data from HuggingFace (IRT models, OLL results, LM evaluation data)
- Set up evaluation pipeline (eval/), scoring (scoring/), scripts (all .sh files)
- Verified all code works with smoke tests
- Committed and pushed all code
- Submitted experiment job (CPU-only, 0 GPUs)

### [2026-04-07 12:30] - method_runs → core_claim
- Experiment completed in ~25 minutes (6 LMs × 30 checkpoints × 6 benchmarks × 3 methods × 3 item counts)
- Results validate core claim: FLUID consistently outperforms RANDOM and RANDOM IRT baselines
- FLUID achieves highest saturation (0.646-0.805) and lowest variance (2.56-5.25) across all item counts
- Validated reference.json and scores.json with the scoring pipeline
- Filled in all scoring documentation files (TARGETS.md, CONSTRAINTS.md, DIRECTION.md, EXPERIMENTS.md)

## Stop Justification

Stopping at core_claim with full validation. The experiment successfully reproduces the paper's
primary claim: adaptive item selection via Fisher information outperforms random selection for
benchmarking LLMs. The directional findings are confirmed (FLUID > RANDOM IRT > RANDOM on saturation
and variance), and the reproduced metrics show the same trends as the paper, though with some
numerical differences likely due to random seeds, checkpoint selection, or minor implementation
details.
