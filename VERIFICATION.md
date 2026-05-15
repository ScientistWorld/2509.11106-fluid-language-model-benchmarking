# Verification Report

## Initial Assessment
The workspace started in a credible `secondary_claims` state. The implementation in `method/`
matches the paper's released-artifact FLUID computation: 2PL ability estimation plus
Fisher-information adaptive item selection. The primary evaluator reads
`checkpoints/experiment_results.json` and writes score artifacts without importing from `method/`,
and the train/test evaluators use the same method-output interface.

Problem/solution separation was mostly clean when verification began. The main checks were whether
the active scores matched the reusable evaluator output, whether the method-safe scoring docs matched
the experiment surface, and whether the download/evaluation scripts remained portable.

## Issues Found
- **Minor**: `scripts/evaluate.sh` failed in the login-node Python because `eval.score_results`
  imported NumPy for simple scoring operations. The packaged container installs NumPy, but the active
  evaluator does not need it and should remain lightweight.
- **Minor**: Regenerating `scoring/scores.json` from the saved method artifact added the Appendix F /
  Table 4 per-LM breakdown that was already present in `reference.json`, but
  `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and parts of `PROGRESS.md` still described the
  main active score surface as Table 2 plus Appendix H only.
- **Residual limitation**: The reproduction uses the authors' released fitted IRT artifacts and
  item-response tables. It implements and evaluates the adaptive benchmarking method, but does not
  reproduce the upstream hierarchical-MCMC fitting stage from raw leaderboard responses.

No critical issues were found in method fidelity, scoring direction, evaluator independence,
train/test separation, or download portability.

## Fixes Applied
- Removed the active scorer's NumPy dependency by rewriting `eval/score_results.py` with
  standard-library rank, mean, total-variation, and Spearman-correlation calculations.
- Made `eval/__init__.py` lazy so running `python -m eval.score_results` does not import optional
  metric-helper dependencies before they are needed.
- Re-ran `scripts/evaluate.sh`, `scripts/evaluate_train.sh`, and `scripts/evaluate_test.sh` against
  `checkpoints/experiment_results.json`; all score files were regenerated successfully.
- Updated `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, `PROGRESS.md`, and `MILESTONES.md` so the
  documentation matches the active Table 2, Appendix H, and Appendix F/Table 4 score surface.
- Verified every committed `scripts/download.sh` fetch target with `curl -I -L --max-time 15`; all
  returned HTTP 200 and no `shared/` paths are referenced by committed method/eval/baseline/scripts
  code.
- Re-ran `python3 validate.py --compare`, Python byte-compilation for tracked Python modules, and
  shell syntax checks for all script entrypoints.

## Final Judgment
The workspace is ready to serve as a research gym at `secondary_claims`. Improve-mode agents have a
working FLUID implementation, reproduced scores, train/test scoring slices, and a packaged
environment. From-scratch agents have method-agnostic problem/evaluation docs, portable data loading,
constraints/direction guidance, and evaluators that score only submitted output artifacts.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: relies on released fitted IRT artifacts rather than reproducing the paper's upstream hierarchical-MCMC fitting stage
