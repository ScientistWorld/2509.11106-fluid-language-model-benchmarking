# Progress

## What Works

- The paper analysis is complete and the core released artifacts are present locally in `data/fluid-benchmarking`.
- The method path now matches the released implementation more closely:
  - MAP ability estimation uses the released Newton-plus-bisection solver.
  - FLUID uses adaptive maximum-Fisher-information item selection.
  - Random baselines now use fixed benchmark-level subsets instead of resampling per checkpoint.
  - FLUID starts from benchmark-specific mean leaderboard ability, matching the released runner.
- Evaluation now lives in `eval/` and converts raw experiment summaries into `scoring/scores.json` without injecting paper numbers.
- The runtime setup now targets a workspace-local dependency directory (`/home/user/.deps/fluid_benchmarking`) because the mounted container root filesystem is only 64 MB and cannot reliably support `--system` installs.
- A smoke test on the released artifacts completed successfully with `python -m method.experiment --item-budgets 10 50 --max-checkpoints 2`, and that payload converts cleanly into the required scoring schema.

## Results

- `method_runs` is reached: the paper's actual FLUID algorithm executes end to end on the released benchmark artifacts.
- `scoring/scores.json` now contains measured smoke-test outputs in the correct experiment-based structure. These are not yet the full Table 2 reproduction.

## Deviations from Paper

- The reproduction uses the paper's released prefit 2PL item parameters and released checkpoint/item-response artifacts rather than re-fitting the 2PL models from raw Open LLM Leaderboard item-level responses. This preserves the paper's evaluation algorithm while reducing compute and data-assembly burden; it should be read as a released-artifact reproduction of Table 2 rather than a full end-to-end reconstruction of the IRT fitting stage.

## Remaining

- Run the full released-artifact experiment and write validated `scoring/scores.json`.
- Update milestones and reference comparison from measured outputs.
- Commit and push before submission.

## Issues

- The login node root filesystem is full, so local validation relies on workspace-local Python packages under `/home/user/.deps/fluid_benchmarking`.
- Full reproduction should run as a submitted CPU job rather than on the login node because it touches all 2,802 checkpoint-benchmark combinations from the released artifacts.
