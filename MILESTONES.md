# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-04-24 03:13] - method_runs
- Verified that the repository implements the paper's actual 2PL IRT ability estimation and Fisher-information adaptive item selection rather than a surrogate.
- Confirmed the method runs end-to-end on the released benchmark artifacts and writes raw experiment outputs for all Table 2 budgets.

### [2026-04-24 03:13] - core_claim
- Re-ran the method-independent evaluator on `checkpoints/experiment_results.json` and regenerated `scoring/scores.json`.
- Confirmed that FLUID beats both RANDOM and RANDOM_IRT on rank distance, total variation, and saturation at budgets 10, 50, 100, and 500, matching the paper's core Table 2 claim.

### [2026-04-24 03:30] - core_claim
- Audited the carried-over workspace and confirmed that `method/` implements the paper's adaptive FLUID stage directly: 2PL MAP ability estimation plus Fisher-information item selection.
- Confirmed that `eval/` remains method-independent and that the stored scores are plausible relative to the paper's Table 2 and Appendix H references.
- Noted the main remaining limitation honestly: the workspace reuses released fitted IRT parameters and released item-response artifacts rather than reproducing the MCMC IRT-fitting stage, so the current closure target remains `core_claim` on released artifacts rather than a fuller milestone.
