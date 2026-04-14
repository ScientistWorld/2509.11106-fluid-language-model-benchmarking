# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-14 00:00] - method_runs
- FLUID's 2PL ability estimation and Fisher-information adaptive item selection are implemented and runnable on the released item-level data.
- Submitted a CPU verification run for the Table 2 core-claim metrics.

### [2026-04-14 02:27] - core_claim
- Job 7d2d5bf5-1b5 completed successfully with exit code 0.
- The end-to-end reproduction ran FLUID, RANDOM_IRT, and RANDOM for budgets 10, 50, 100, and 500.
- FLUID beat both baselines on rank distance, total variation, and saturation at every evaluated budget.
- `scripts/evaluate.sh` wrote `scoring/scores.json`, and `validate.py --compare` passed.

### [2026-04-14 02:45] - core_claim
- Extended the local scoring target to include Appendix H's full-benchmark accuracy comparison.
- Local evaluation shows `fluid_500` beating `full_accuracy` on rank distance, total variation, and saturation.
- Submitting a CPU test job to verify the updated evaluator in the compute environment before raising the milestone.

### [2026-04-14 03:10] - core_claim
- Final verification kept the milestone at `core_claim`.
- `scripts/evaluate.sh` passed locally after scoring and documentation fixes.
- Method-agnostic docs were tightened for from-scratch gym use, and generated package/upstream cache files were removed from the tracked root layout.

### [2026-04-14 02:55] - core_claim
- Added `scripts/baseline.sh` as a wrapper for the matched RANDOM and RANDOM_IRT Table 2 baselines.
- Documented that this reproduction uses the authors' released fitted 2PL item parameters rather than re-running hierarchical MCMC IRT fitting.
- Final verifier re-ran `scripts/evaluate.sh /home/user/checkpoints/experiment_results.json`; validation passed and regenerated `scoring/scores.json`.
- Removed method identifiers from strictly method-agnostic experiment/target docs and removed an unused `tmp_apt/` scratch directory.

## Stop Justification
- Completed at milestone `core_claim`.
