# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-07 10:30] - none
- Read the paper and identified key components
- Paper proposes FLUID BENCHMARKING combining IRT with adaptive item selection
- Core algorithm: 2PL IRT model + Fisher information-based item selection
- Data available: pre-fitted IRT models, OLL results, 6 test LM checkpoint evaluations

### [2026-04-07 11:00] - none → method_runs (preparation)
- Filled in all briefing files (problem, evaluation, method, overview)
- Implemented IRT model: ability estimation, Fisher information, logistic function
- Implemented FluidBenchmarking class with adaptive item selection
- Implemented all 4 evaluation metrics (validity, variance, saturation, efficiency)
- Downloaded all data from HuggingFace (IRT models, OLL results, LM evaluation data)
- Set up evaluation pipeline (eval/), scoring (scoring/), scripts (all .sh files)
- Verified all code works with smoke tests
- Committed and pushed all code
- Submitted experiment job (CPU-only, 0 GPUs)

## Stop Justification

<!-- Do not edit this unless you decide to stop -->

<!-- Write a brief explanation of WHY you are stopping at this milestone.
     What blocked you to reach the next milestone and you decided to stop?
     This is NOT a progress summary — the Progress Log above covers that.
     Example: "Stopping at core_claim because the full dataset requires
     proprietary access. Core claim validated on the public subset." -->