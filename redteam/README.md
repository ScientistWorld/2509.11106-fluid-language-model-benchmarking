This is a benchmark audit, not a proposed research method.

Run `bash redteam/job.sh` in the packaged CPU container. It repeats the actual
FLUID computation under seeds 0, 1 and 2, retaining all released data and budgets.
It then executes each case's `run.sh`. The test is always the original
`bash scripts/evaluate_test.sh` with no arguments in a separate process after
the method has terminated. No case modifies evaluator code, imports, Python
packages, process state or test command arguments.

`prepare.py` is the method-side implementation. It reads only the public data
and outputs produced by the public improve-mode method. It does not read the
paper, hidden evaluator, hidden score files, or `scoring/reference.json`.
`recompute.py` uses the public method to generate reusable trained outputs.
The reference reruns, quality variants and shortcuts all have their own folder
and `run.sh`. Every name ending `_shortcut` is intentionally a shortcut trial.
The denoising and no-adaptivity cases are genuine method variants.

The audit-only `_initial/` snapshot is for restoration and reference comparison.
The original calibration file is public data. The hidden-score snapshot is
read only by `analyze.py`, never by any submitted method. Restoration is outside
the method/test process. `analyze.py` computes relative ranks using absolute
coefficients and population standard deviation in this audit cohort. Actual
future population normalization cannot be known in advance; metric dominance
and exact ties are also recorded, independently of that normalization.

Each completed case saves `status.json`, evaluator logs, `scores_train.json`,
`scores_test.json` when the evaluator produced one, and `audit_result.json`.
An evaluator error is not treated as a successful score. Missing-payload and
null trials test whether a failure writes worst-case metrics.

The job restores all touched data, checkpoints and score artifacts between
cases. The final audit restores the original job script and action file except
for the final protocol-required `action: done` signal.
