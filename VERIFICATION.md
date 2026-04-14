# Verification Report

## Initial Assessment
The workspace started at `core_claim`, with a completed CPU reproduction of the main Table 2
comparison and a locally validated extension for the full-benchmark accuracy comparison. The
claimed milestone is justified: the implementation runs the paper's actual benchmark-estimation
algorithm, and the reproduced results show the candidate method beating both subset baselines on
rank distance, total variation, and saturation across the evaluated budgets.

## Issues Found
- **Minor**: Several from-scratch-visible docs described the paper's specific mechanism instead of
  only the problem, score contract, and allowed evaluation scope.
- **Minor**: The `rank_distance` coefficient in `scoring/reference.json` was positive even though
  higher values are worse; that would reward the wrong direction if a downstream scorer used the
  coefficient sign directly.
- **Minor**: Local package-cache files and an upstream checkout were tracked or located at the
  workspace root, which made the root layout noisier than needed for a reusable gym.

## Fixes Applied
- Rewrote `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`,
  `scoring/CONSTRAINTS.md`, `scoring/DIRECTION.md`, and `scoring/reference.json` to describe the
  problem and experiments without exposing the paper's method in from-scratch-visible text.
- Changed `rank_distance` coefficients to `-1.0` for both reproduced experiments while keeping
  `total_variation` negative and `saturation` positive.
- Untracked `.pkgs/` and the root upstream checkout, moved the local upstream checkout under
  `tmp/`, and removed an unused validator backup from the root.
- Re-ran `bash scripts/evaluate.sh /home/user/checkpoints/experiment_results.json`; validation
  passed and `eval/`/`data/` import separation remains intact.

## Final Judgment
The workspace is ready to serve as a gym. Improve-mode agents can use the method implementation and
reproduced scores directly. From-scratch-mode agents can use the problem/evaluation/scoring docs,
data, and reusable evaluator without being handed the paper's specific algorithm in the
method-agnostic files. The main limitation is scope: it verifies the core Table 2 claim and a
secondary full-accuracy comparison, not every paper table.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: core and secondary comparisons reproduced, but broader baseline families from the paper are not fully reproduced.
