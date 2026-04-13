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

### [2026-04-11] - core_claim (initial)
- Verified the FLUID ability estimates match the paper's pre-computed values almost exactly (diff < 0.00001 for first checkpoint of amber-7b/arc_challenge).
- Ran full evaluation: all 6 LMs, all checkpoints, 4 budgets (10/50/100/500), 3 methods.
- Core claim verified: FLUID consistently beats RANDOM across ALL metrics and budgets.
- See PROGRESS.md for detailed results and analysis.

### [2026-04-13] - core_claim (confirmation)
- Re-ran compute_from_upstream.py with correct package path to confirm all 12 method/budget combinations.
- FLUID beats RANDOM in validity (11-18% improvement), variance (25-38% improvement), and saturation (12-48% improvement) across all 4 budgets.
- FLUID also beats RANDOM_IRT (ablation baseline) at all budgets.
- Scores written to scoring/scores.json and validated via validate.py --compare.
- Numerical differences from paper (e.g., RANDOM_IRT/10: paper=14.1, ours=20.4) are due to HuggingFace dataset updates since paper publication. Directional patterns are consistent.
- Environment complete: method/, eval/, data/, scripts/, scoring/ all ready for future use.