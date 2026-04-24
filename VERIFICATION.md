# Verification Report

## Initial Assessment
The workspace started in a mostly usable `secondary_claims` state. The core FLUID reproduction was intact, `validate.py` passed, the method/eval separation was clean, and the packaged environment had already been validated on-cluster. The main issues were gym-facing consistency problems: stale root-level environment duplicates, scoring docs that described an unreproduced Appendix F experiment as if it were active, and metric text in `reference.json` that did not match the implemented saturation computation.

## Issues Found
- **Minor**: The workspace root still contained duplicate `container.def` and `setup.sh` files even though the packaged deliverables live under `environment/`.
- **Minor**: `scoring/EXPERIMENTS.md` and `scoring/TARGETS.md` described `appendix_f_table4_lm_breakdown_100`, but that experiment is not present in `scoring/scores.json`.
- **Minor**: `PROGRESS.md`, `CHECKLIST.md`, and `MILESTONES.md` still implied the root-level setup shim was part of the final packaged layout.
- **Minor**: The `saturation` metric descriptions in `scoring/reference.json` said Pearson correlation, while the scorer computes an absolute Spearman-style rank correlation.
- **Minor**: The environment packaged Python dependencies through `apt` in `environment/container.def`, which conflicts with the repo rule that Python packages belong in `environment/setup.sh`.

## Fixes Applied
- Removed the duplicate root-level `container.def` and `setup.sh` so the environment entrypoints live only in `environment/`.
- Aligned `scoring/EXPERIMENTS.md` and `scoring/TARGETS.md` with the experiments actually present in `scoring/scores.json` (Table 2 and Appendix H).
- Updated `PROGRESS.md`, `CHECKLIST.md`, and `MILESTONES.md` to reflect the cleaned packaging layout and to describe Appendix F as reference coverage rather than an active reproduced score target.
- Corrected the `saturation` descriptions in `scoring/reference.json` to match the implemented absolute Spearman-style computation.
- Moved `numpy`, `pandas`, and `scipy` installation out of `environment/container.def` and into `environment/setup.sh` via `uv pip install --system`.
- Re-verified `scripts/download.sh` URLs; every committed fetch target resolves successfully (HTTP 200) from the login node.
- Removed the ignored local `environment/container.sif` build artifact. The leftover `environment/container.final.def` path is a root-owned special file excluded from git, so it does not affect the shipped workspace.
- Re-ran `python3 validate.py` and confirmed the structure/import checks still pass.

## Final Judgment
The workspace is ready to serve as a gym at the claimed `secondary_claims` milestone. Improve-mode agents have a working FLUID implementation, packaged environment, and reproduced Table 2 plus Appendix H score surface. From-scratch agents have method-agnostic problem and evaluation docs, reusable data access, and an evaluator that does not import from `method/`. The main remaining limitation is scope of reproduction provenance rather than score coverage: the workspace still relies on the released fitted IRT artifacts instead of reproducing the upstream hierarchical-MCMC fitting stage from scratch.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: the reproduction still depends on the released fitted IRT artifacts and does not reproduce the paper's upstream hierarchical-MCMC fitting stage

## Evaluation Split Outcome

The train/test evaluation pass completed successfully and the split is usable.

- `scoring/scores_train.json` and `scoring/scores_test.json` both exist and contain real values for all three split experiments.
- Result identifier sets match exactly between train and test for every experiment.
- The split is non-trivial rather than byte-identical: average absolute train-test metric gap is `2.861` for Table 2, `2.736` for Appendix H, and `1.287` for Table 4.
- `fluid` remains better than matched baselines on all 24 Table 2 metric comparisons in both slices and on all 36 Table 4 metric comparisons in both slices when using metric coefficient direction.
- The Table 4 split was recalibrated from a parity split to a deterministic hash-balanced checkpoint split, which reduced its average train-test gap while preserving row coverage and ranking signal.

The main residual risk is calibration of `total_variation`: a few held-out rows are much harsher than train, especially `random_irt_50` in Table 2, `random_amber_7b_100` / `random_irt_pythia_7b_100` in Table 4, and `full_accuracy` in Appendix H. That is a caveat, not a blocker.
