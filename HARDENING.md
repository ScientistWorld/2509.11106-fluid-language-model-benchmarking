# Hardening report — FLUID benchmarking gym (iteration 11)

**Bottom line.** The hidden test now rewards better measurement. An honest better variant beats the reference by more than paired noise on the hidden test: rank +0.216, 95% bootstrap CI [+0.154, +0.278], above the reference in 100% of resamples. Before, the better variant's gain sat inside a 3.75 seed spread. Every major shortcut from `REDTEAM.md` is now either impossible by construction or scored as a failed run (worst value in every metric). The reproduction also became more faithful. Restoring the paper's released response binarization makes the hidden-test per-LM values match the paper's Table 4 to within 0.05. Pooled over all six LMs, validity is 10.12 / 8.78 / 8.65 / 8.27 against the paper's 10.1 / 8.8 / 8.7 / 8.3.

All evidence is in `hardening/` (`run_audit.sh`, `probes/`, `analyze.py`, `cases/<case>/`, `summary.json`, `summary.md`). It was produced in jobs `c6167f00-2ce` and `69bcfe1b-31d`. Every case ran the unchanged `bash scripts/evaluate_test.sh` (or `evaluate_train.sh`) with no arguments, with the probe installed as `scripts/method.sh`. The workspace was restored afterwards.

## 1. What changed

### Evaluator-run item-query protocol (the central fix)
Before, the method wrote a JSON payload of score curves, ground-truth accuracy curves and baseline curves. The evaluator trusted it, and also trusted a mutable calibration file in `data/`. Now:

- **The evaluator runs the method itself.** The submission is the executable `scripts/method.sh`. `eval/test/score.py` starts it as a subprocess and runs one session per (held-out checkpoint, benchmark, budget): 4,848 sessions on test and 6,000 on train. In a session the method sends `Q <item>` up to `budget` times and receives `R 0|1`, then returns `S <score>`. It sees only the answers it queried, and never the LM or checkpoint identity. Sessions arrive shuffled across LMs, checkpoints, benchmarks and budgets. The protocol is in `briefing/evaluation.md`. The reference implementation is `method/serve.py`, which wraps the existing `method/irt.py` and `method/fluid.py`.
- **Everything on the comparison side is fixed in `eval/`.**
  - Item budgets and the checkpoint roster.
  - Ground truth: full-benchmark accuracy, computed after the method returns.
  - The 102-LM reference population: a pinned copy at `eval/*/data/oll_reference.json`, checked by sha256.
  - Baselines: RANDOM, RANDOM IRT, full-benchmark accuracy and all-item ability, computed by the evaluator. RANDOM and RANDOM IRT are averaged over five fixed subsets. All baselines are reported unranked under `baselines`.
- **Every scored number is recomputed from the raw per-session scores.** Nothing the method writes is read except those scores.
- **Hidden data.** The held-out LMs' responses (K2-65B, Pythia-2.8B, Pythia-6.9B) now live only in `eval/test/data/hidden_responses.json`: binarized, bit-packed and sha-pinned. The file is chmod 000 while the method runs. They were removed from `data/downloads/`. I also deleted about 40 MB of old redteam payloads that contained these LMs' ground-truth accuracy curves. `scripts/download.sh` pins the Hugging Face revision `e0a11a3` and checks every file's sha256. It rebuilds the hidden table byte-identically from the public release, staging the held-out CSVs in a temp directory only.
- **Fail-closed scoring.** The following fail the run:
  - over-budget or repeated queries, or out-of-range item indices;
  - non-finite scores, accuracy outside [0, 1], or ability with |value| > 1e6;
  - a crash, or 900 s of silence;
  - any missing session;
  - a replay mismatch;
  - modification of the public data files.

  A failed run is written to `scores_*.json` with **every** scored metric at its worst value (rank distance 102, TV 100, saturation 0), `"status": "failed"`, and exit code 0, so the file is always ranked.
- **Statelessness and determinism.** A random 12% of sessions are first replayed in a fresh method process, in a different order. They must give identical queries and scores to the main pass. This rules out pooling or smoothing across checkpoints. The paper evaluates each checkpoint on its own, and its variance and saturation metrics measure the noise of that per-checkpoint evaluation.
- **Degenerate curves.** A constant curve, or one whose last value equals its first, now scores the worst TV (100) instead of the best (0). Eq. 6 is undefined there and the curve carries no signal. No reference curve is affected.
- **No method name hardcoded.** The scored rows are `submitted_<budget>`; the paper's FLUID values are kept unchanged in `reference.json` under those keys. `scripts/evaluate_test.sh` and `evaluate_train.sh` take no arguments and score whatever `scripts/method.sh` serves. `eval/train/score.py` and `eval/test/score.py` share no code, and neither imports `method/`.

### Restoring the paper's own settings
- **Binarization.** The paper's released loader (`datasets.load_lm_eval_results(binary=True)`) scores an item correct when its value is at least 0.5. The gym had been feeding TruthfulQA-MC2's fractional scores straight into the likelihood and into accuracy. Both evaluators and `data/loaders.py` now binarize at 0.5.
- **Already matching.** The reference already matched the official code: MAP with a N(0,1) prior on [−4, 4], Newton steps with bisection fallback, start ability at the mean reference-LM ability, and a fixed random subset per benchmark for the ablations.

### Experiments and weights
- `table2_ablation_comparison` (0.65): rows `submitted_10/50/100/500`.
- `appendix_h_full_accuracy_comparison` (0.15): row `submitted_500`.
- `appendix_f_table4_worst_lm_100` (0.20) replaces the per-LM rows. The held-out LMs can no longer share row names with the visible ones, and train and test must have identical keys. Per-LM consistency is therefore scored as the slice's worst LM at 100 items. `results` hold paper-derived worst-LM values; the original per-LM Table 4 numbers are kept under `_paper_table4_per_lm_100_items`.
- Weights and coefficients are unchanged (−1 rank distance, −0.8 TV, +1 saturation).

## 2. Do better results score higher? (hidden test)

Ranks follow the gym's rule: the sum over scored cells of weight × |coefficient| × improvement in its better direction, divided by that cell's standard deviation across attempts. The future cohort is unknown, so two normalizations are shown. The first uses all successful audit attempts, including the degenerate constant output. The second uses only honest method variants (reference, better, EAP-final, predicted-accuracy, train-tuned shift, weaker). The 95% CIs are paired bootstrap intervals (2,000 resamples of checkpoints and benchmark curves within each LM, with both methods seeing the same resample).

| Variant | What it changes | Before (old gym) | After: rank (all) | After: rank (honest) | 95% CI (all) | Cells + / − |
|---|---|---:|---:|---:|---|---|
| Reference (FLUID) | — | 0 | 0 | 0 | — | 0 / 0 |
| **Better: same-benchmark calibration** | Same FLUID session, then a per-benchmark, per-budget affine map from the short adaptive estimate to the same benchmark's all-item ability, fitted only on the visible LMs (`probes/fit_calibration.py`) | — | **+0.216** | **+0.889** | **[+0.154, +0.278]** | 4 / 2 |
| Old "better denoising" | 3-point smoother across checkpoints | +0.939 (inside the ±3.75 seed spread) | not expressible; its stateful analogue fails | — | — | — |
| EAP final aggregation | Posterior mean instead of mode | — | −0.004 | +0.006 | [−0.042, +0.043] | 6 / 10 (neutral) |
| Predicted-accuracy scale | Same session, reported as 2PL-predicted full-benchmark accuracy | — | −0.903 | −4.544 | [−1.07, −0.73] | 0 / 12 |
| **Weaker: no adaptivity** | Random subset + MAP (RANDOM IRT served as the method) | −3.780 | **−4.795** | **−19.21** | [−6.11, −3.70] | 0 / 18 |

The better variant's hidden-test validity (rank distance) moves 9.444→8.738 (10 items), 8.077→7.702 (50), 8.078→7.627 (100) and 7.616→7.635 (500, CI includes 0). Worst-LM validity moves 11.63→10.71. TV and saturation are unchanged by construction. As independent evidence that it is a better measurement, its RMSE to the same benchmark's all-item ability falls at every budget: 0.494→0.470, 0.443→0.414, 0.443→0.398, 0.389→0.343. On train (in-sample fit) it improves 6 cells and worsens none.

The predicted-accuracy variant improves all 12 non-tied cells on the *visible* LMs but loses on all 12 held-out cells. The held-out-LM split is what separates genuine improvements from train-specific ones.

**Seed noise.** The reference is deterministic. Its rerun differs from the first run by 0.0 in every scored cell, and the replay check confirms it within each run. Baselines are evaluator-owned (five-subset means) and unranked, so the complete rank no longer has the 3.75 seed spread the redteam found, which came entirely from method-written baseline rows.

## 3. Major shortcuts: fix and score before → after

"Before" ranks are from `REDTEAM.md` (old cohort and normalization). "After" ranks use the all-successful cohort above; failed runs score −66.76 there.

| Shortcut (REDTEAM) | Before | Fix | After (probe in `hardening/cases/`) |
|---|---:|---|---|
| Fabricated curves and ground truth | +50.18 | Ground truth is computed by the evaluator from hidden responses; the method returns only per-session scores | Not expressible |
| Replace calibration file | +19.60 | Evaluator-owned, sha-pinned reference population; any change to `data/downloads` fails the run | `tamper_public_data`: **failed**, worst everywhere |
| Baseline rows written by the method | +9.69 | Baselines computed by the evaluator and unranked | Not expressible |
| Constant output | +4.14 | No baseline bonus; flat curves get worst TV and saturation | `constant_output`: −26.32, 0 / 18 cells |
| Short test (fewer checkpoints) | +49.37 | Roster fixed by the evaluator; every session must be answered | Not expressible; skipping sessions fails (`crash_on_budget_500`: **failed**) |
| Read target data (rank distance 0.07) | +0.15 | Held-out LMs only in `eval/test`, unreadable during the run; a session exposes only queried answers of one benchmark | `read_hidden_data`: identical to reference (0.000); file read raised `PermissionError` |
| Omit result rows | 0 (tie) | Fixed roster | `crash_on_budget_500`: **failed** |
| Unlisted failed method | 0 (tie) | One submission; every session scored | Not expressible |
| All-items brute force (budget accepted) | −0.26 | Query counter enforces the budget | `over_budget_bruteforce`: **failed** |
| Forged score fields | −8.31 | Method writes no score files | Not expressible |
| Empty / singleton / NaN / null / missing payloads | −8.6 / −8.6 / −23.7 / no score / no score | Impossible values fail; failures still write worst values | `nan_output`, `accuracy_out_of_range`, `missing_method`: **failed**, all with full worst-value score files |
| Cross-checkpoint smoothing | +0.94 | Replay check (stateless per-session procedure) | `stateful_smoothing`: **failed** (582/582 replayed sessions differ) |
| Recalibrating toward the paired benchmark, tuned on visible LMs (new probe) | — | Held-out-LM split | `train_tuned_shift`: −1.12; it improves train validity but loses 6/6 validity cells on test |

## 4. Shortcuts left to the grader, and why
- **Reading `eval/test/` as root.** File permissions do not bind root. Jobs run as an unprivileged user (`uid=374202`), and the probe was refused. A method that reads `eval/` breaks `scoring/CONSTRAINTS.md`.
- **Embedding the held-out LMs' public responses** fetched from Hugging Face into the method. The data is public, so no technical barrier is possible; this is forbidden in `CONSTRAINTS.md`.
- **Cross-session state hidden from the replay check**, e.g. caching outputs to disk during the replay pass and reusing them in the main pass. Defeating this requires deliberately targeting the evaluator. The replay pass runs first, in a fresh process and a different order, which catches ordinary stateful methods.
- **Scale and calibration choices.** The validity rank depends on the declared scale and calibration. These are legitimate method choices, and the held-out split punishes train-specific ones (`train_tuned_shift`, `predicted_accuracy_scale`).

## 5. Reference score before → after (hidden test)

| Row | Before (rank distance, TV, saturation) | After |
|---|---|---|
| 10 items | (9.448, 13.591, 0.7030) | (9.444, 16.081, 0.6941) |
| 50 items | (8.088, 6.919, 0.8483) | (8.077, 7.018, 0.8486) |
| 100 items | (8.092, 6.238, 0.8512) | (8.078, 6.048, 0.8577) |
| 500 items (also Appendix H row) | (7.615, 4.867, 0.8777) | (7.616, 4.924, 0.8798) |
| Per-LM experiment | 18 per-LM rows | worst LM at 100 items: (11.628, 7.139, 0.813) |

The changes come from binarizing TruthfulQA. At 10 items, K2-65B's binarized TruthfulQA curve ends almost where it starts, so the existing TV cap of 100 applies to it. Train reference: (10.670, 12.505, 0.722), (9.345, 6.875, 0.814), (9.121, 6.067, 0.842), (8.794, 4.950, 0.871); worst LM (19.274, 6.828, 0.802).

**Paper direction.**
- Hidden test: the reference beats both evaluator baselines on 23/24 Table 2 comparisons. The exception is RANDOM IRT validity at 500 items, also the paper's closest gap (8.4 vs 8.3). It beats full-benchmark accuracy on 3/3 Appendix H metrics and on 18/18 per-LM comparisons.
- Train: 23/24, 3/3 and 17/18. The exception is Amber-6.7B validity against RANDOM IRT.
- Pooled over all six LMs (`hardening/summary.json: six_lm_reference`):

| | 10 | 50 | 100 | 500 items |
|---|---|---|---|---|
| Validity (paper) | 10.12 (10.1) | 8.78 (8.8) | 8.65 (8.7) | 8.27 (8.3) |
| TV (paper) | 14.3 (10.7) | 6.9 (6.5) | 6.1 (6.1) | 4.9 (4.9) |
| Saturation (paper) | 0.71 (0.76) | 0.83 (0.86) | 0.85 (0.85) | 0.875 (0.88) |

- Hidden-test per-LM values against Table 4: K2 (2.07, 7.14, 0.892) vs (2.1, 7.1, 0.89); Pythia-2.8B (8.07, 6.55, 0.813) vs (8.1, 6.5, 0.81); Pythia-6.9B (11.63, 4.46, 0.868) vs (11.6, 4.5, 0.87).

## 6. Headroom check
- **No reference cell sits at a bound.** The bounds are rank distance 0, TV ≈ n/(n−1) ≈ 1 for a monotone curve, and saturation 1 (`summary.json: reference_cells_at_bound = []`).
- **A variant scores above the reference** by more than paired noise (section 2), so no extra experiment was needed.
- **Room remains above it.**
  - All-item ability reaches validity 7.10 on test, against 9.44 (10 items) to 7.62 (500).
  - The reference's TV at 10 items (16.1) is far above its 500-item value (4.9).
  - Of everything tried, only the calibration transferred. Neutral or worse on the held-out LMs: EAP, posterior-weighted information selection, empirical and wider priors, lapse-robust likelihoods, discrimination caps, item-misfit filtering, refitting item parameters on the visible LMs, and dynamic stopping at matched mean budget. These were prototyped on the visible LMs; only the two variants in section 2 were run through the hidden test.
- **Brute force** (all items) cannot be submitted, because the budget is enforced. As a baseline, all-item ability does not dominate the reference: on test its validity is better (7.10) but its TV is worse (15.66 vs 4.92 at 500 items).
- **Simple baselines sit clearly below the reference.** Hidden test, 10 / 500 items (rank distance, TV): RANDOM (25.1, 43.8) / (10.1, 15.6) and RANDOM IRT (17.4, 39.8) / (7.36, 13.8), against the reference's (9.44, 16.1) / (7.62, 4.92). The scale also leaves room above the reference (section 2).

## 7. What I could not fix
- **IRT fitting is not re-run.** The hierarchical-prior MCMC fit on the 102 reference LMs is not reproduced; the gym uses the paper's released item parameters and responses. The released K2 table has 46 checkpoints (paper: 61).
- **Other Table 1 baselines are absent** (Anchor Points, tinyBenchmarks, metabench, SMART, MAGI).
- **The paper's direction misses in two places.** Amber-6.7B validity loses to RANDOM IRT on train. RANDOM IRT edges FLUID on validity at 500 items in both slices.
- **TV is heavy-tailed.** A near-flat curve (K2 TruthfulQA at 10 items) hits the cap of 100 and dominates `submitted_10` TV. This is the paper's metric on the paper's data; I did not change the aggregation.
- **The better variant's margin is modest** (validity only; +0.4 to +0.7 rank units at ≤ 100 items, neutral at 500). The honest frontier above FLUID on these metrics appears narrow for per-checkpoint procedures.
- **`validate.py` warning.** It prints a soft "eval looks detached" warning for both evaluators. This is a false positive: the evaluators read the method's outputs through the protocol, not from a file.
- **The old redteam scripts in `redteam/` target the removed payload interface.** They are kept as the historical record of the previous gym.
