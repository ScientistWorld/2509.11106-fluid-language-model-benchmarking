# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-11] - none
- Read the paper thoroughly. FLUID BENCHMARKING combines 2PL Item Response Theory with adaptive item selection using Fisher information to improve LM evaluation across four dimensions: efficiency, validity, variance, and saturation.
- Previous agent set up the workspace: briefing files, method code, eval code, data from HuggingFace allenai/fluid-benchmarking, container definition, and scripts.
- Previous agent reached `method_runs` via smoke test (2 budgets, 3 checkpoints per LM).

### [2026-04-11] - method_runs
- Fixed PYTHONPATH in all shell scripts (was missing `/home/user/.deps/fluid_benchmarking`).
- Verified full pipeline works on login node (80s for smoke test).
- Submitted full reproduction: all 6 LMs × all checkpoints (61-94 each) × 6 benchmarks × 4 budgets × 3 methods.
- Expected runtime ~1 hour.

### [2026-04-11] - core_claim (pending)
- Awaiting full experiment results. Will compare reproduced metrics against paper's Table 2 numbers.
