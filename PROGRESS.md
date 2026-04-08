# Progress

## What Works

- The paper analysis is complete and the core released artifacts are present locally in `data/fluid-benchmarking`.
- The method path now matches the released implementation more closely:
  - MAP ability estimation uses the released Newton-plus-bisection solver.
  - FLUID uses adaptive maximum-Fisher-information item selection.
  - Random baselines now use fixed benchmark-level subsets instead of resampling per checkpoint.
  - FLUID starts from benchmark-specific mean leaderboard ability, matching the released runner.
- Evaluation now lives in `eval/` and converts raw experiment summaries into `scoring/scores.json` without injecting paper numbers.

## Results

- No new valid reproduction numbers have been claimed in this turn yet.
- Previous `scores.json` contents are being treated as untrusted until the corrected pipeline is re-run and validated.

## Remaining

- Install runtime dependencies in the environment used for local testing and jobs.
- Run a smoke test to reach `method_runs` with the corrected pipeline.
- Run the full released-artifact experiment and write validated `scoring/scores.json`.
- Update milestones and reference comparison from measured outputs.
- Commit and push before submission.

## Issues

- Local Python on the login node currently lacks required packages such as `numpy`, so local smoke tests need dependency installation first.
- The previous evaluation script was not valid for reproduction because it copied paper validity numbers into the computed score file.
