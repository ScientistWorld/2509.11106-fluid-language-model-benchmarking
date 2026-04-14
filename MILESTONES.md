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

## Stop Justification
- Current verified compute milestone is `core_claim`; Appendix H extension is pending compute-job verification.
