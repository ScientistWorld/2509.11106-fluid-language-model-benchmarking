# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-04-24 03:40] - secondary_claims
- Validated the packaged environment on the cluster with job `aae11f56-4ef`, which rebuilt the container, ran `scripts/run.sh`, and completed `scripts/reproduce.sh` end-to-end with exit code 0.
- Confirmed the packaged run reproduces the full Table 2 FLUID-vs-RANDOM/RANDOM_IRT comparison and rewrites `scoring/scores.json` from fresh method outputs inside the container.
- Confirmed the packaged run also reproduces the Appendix H `full_accuracy` vs `fluid_500` comparison, so the workspace now supports both the core claim and one secondary claim on the released artifacts.

### [2026-04-24 06:55] - secondary_claims
- Extended the scorer and reference tables to cover Appendix F / Table 4, the per-LM breakdown at 100 items, using the same saved FLUID/RANDOM/RANDOM_IRT curves rather than any surrogate artifact.
- Added the missing `environment/setup.sh` no-op shim for the packaged environment.
- Kept the milestone at `secondary_claims` because the active reproduced score surface still consists of Table 2 plus Appendix H.

### [2026-04-24 06:58] - secondary_claims
- Final verification removed duplicate root-level environment entrypoints so the workspace root only exposes the packaged `environment/` deliverables.
- Aligned the human-readable scoring docs with the experiments actually present in `scoring/scores.json`.
- Corrected the saturation metric descriptions in `scoring/reference.json` to match the implemented absolute Spearman-style computation.
