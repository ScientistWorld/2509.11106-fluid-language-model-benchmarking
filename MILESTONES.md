# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-24 02:40] - core_claim (restored)
- Full audit completed: workspace is intact and trustworthy
- Implementation verified: 2PL IRT (irt.py), Fisher-information adaptive selection (fluid.py), metrics (metrics.py)
- Evaluation verified: eval/ is independent from method/, scores are real (from running code)
- All FLUID results match paper within ~1-2%
- Core claim confirmed: FLUID beats RANDOM and RANDOM_IRT on validity, variance, saturation at all budgets
- Appendix H verified: FLUID/500 outperforms full-benchmark accuracy
- All validation checks pass (structure, import separation, scoring)
- Environment is fully reusable for future agents

### [2026-04-24 02:30] - core_claim
- Full reproduction complete: FLUID method runs end-to-end on all budgets
- FLUID/50, FLUID/100, FLUID/500 match paper values closely (within ~1%)
- Core claim confirmed: FLUID beats RANDOM and RANDOM_IRT on all three metrics
- Appendix H secondary claim verified: FLUID/500 outperforms full-benchmark accuracy
- All validation checks pass

### [2026-04-15 06:00] - core_claim
- First successful job run with Table 2 results
- Implemented 2PL IRT (irt.py), Fisher-information selection (fluid.py), metrics (metrics.py)
- Results align with paper's core claims

### [2026-04-09 11:48] - method_runs
- Paper implementation in method/: irt.py, fluid.py, metrics.py, experiment.py
- Evaluation code in eval/ (independent from method/)
- Data loaded from HuggingFace allenai/fluid-benchmarking artifacts

## Stop Justification

