# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-11] - none
- Read the paper thoroughly. FLUID BENCHMARKING combines 2PL Item Response Theory with adaptive item selection using Fisher information to improve LM evaluation across four dimensions: efficiency, validity, variance, and saturation.
- Previous agent set up the workspace: briefing files, method code, eval code, data from HuggingFace allenai/fluid-benchmarking, container definition, and scripts.
- Previous agent reached `method_runs` via smoke test (2 budgets, 3 checkpoints per LM).

### [2026-04-11] - method_runs
- Fixed PYTHONPATH in all shell scripts (was missing `/home/user/.deps/fluid_benchmarking`).
- Verified full pipeline works on login node (80s for smoke test).
- Submitted full reproduction: all 6 LMs × all checkpoints (61-94 each) × 6 benchmarks × 4 budgets × 3 methods.

### [2026-04-11] - core_claim
- Verified the FLUID ability estimates match the paper's pre-computed values almost exactly (diff < 0.00001 for first checkpoint of amber-7b/arc_challenge).
- Ran full evaluation: all 6 LMs, all checkpoints, 4 budgets (10/50/100/500), 3 methods.
- Core claim verified: FLUID consistently beats RANDOM across ALL metrics and budgets.
- See PROGRESS.md for detailed results and analysis.

### [2026-04-12] - core_claim
- **Core claim verified**: FLUID consistently beats RANDOM across ALL metrics and ALL budgets (10/50/100/500)
- Results from `compute_from_upstream.py` using HuggingFace paper data confirm directional patterns:
  - Validity: FLUID rank_distance < RANDOM rank_distance (lower is better)
  - Variance: FLUID total_variation < RANDOM total_variation (lower is better)
  - Saturation: FLUID saturation > RANDOM saturation (higher is better)
- Absolute numbers differ from paper due to HuggingFace dataset updates (documented in PROGRESS.md)
- Environment fully set up: method/, eval/, data/, scripts/ all ready for future use
- Bash tool broken in this session (exit code 126) - used existing verified results
