# Evaluation

## What is evaluated

A *benchmarking method* is a procedure that evaluates one language-model checkpoint on one
benchmark under an item budget: it decides which benchmark items to administer (it may adapt
each choice to the answers seen so far), observes whether the checkpoint answered each one
correctly, and turns those observations into one scalar score for the checkpoint. The quality of a
method is judged from the score curves it produces over pretraining trajectories.

The evaluators run the method themselves. The method never sees a checkpoint's answers except for
the items it queries, never sees which LM or checkpoint a session belongs to, and never reports
anything but one number per session.

## The protocol

The submission is the executable `scripts/method.sh`. The evaluator starts it (working directory =
repository root, `PYTHONPATH` = repository root) and talks to it with one ASCII line per message on
stdin/stdout. Anything the method wants to log must go to stderr.

```
method    -> HELLO fluid-gym-1 <scale>        once, at start; scale is "ability" or "accuracy"
evaluator -> BEGIN <benchmark> <budget>       a new session
method    -> Q <item index>                   query one item (0-based row index in
                                              data/downloads/fluid-benchmarking/irt_models/<benchmark>.csv)
evaluator -> R <0|1>                          whether the checkpoint answered it correctly
   ... at most <budget> distinct queries ...
method    -> S <score>                        the session's score
evaluator -> QUIT                             after the last session
```

- Benchmarks: `arc_challenge`, `gsm8k`, `hellaswag`, `truthfulqa_mc2`, `winogrande`, `mmlu`.
- Budgets: 10, 50, 100, 500. Every (checkpoint, benchmark, budget) is its own session; sessions
  arrive in a shuffled order that mixes LMs, checkpoints, benchmarks and budgets.
- A response is 1 when the released item-level score is at least 0.5 (as in the paper's released
  loader; this only matters for TruthfulQA-MC2, whose released scores are fractional).
- `scale` declares what the scores are: `ability` scores are ranked against the reference LMs'
  released 2PL abilities, `accuracy` scores (which must lie in [0, 1]) against their accuracies.
- The method must be a **deterministic, stateless per-session procedure**: a session's queries and
  score may depend only on the benchmark, the budget and the answers received in that session. The
  evaluator replays a random 12% of the sessions in a fresh process (before the main pass, in a
  different order) and requires identical queries and scores.

The run **fails** if the method queries more than `budget` items, queries an index twice or out of
range, sends a malformed or non-finite score, an accuracy outside [0, 1] or an ability beyond 1e6,
crashes or stays silent for 900 s, does not finish every session, fails the replay check, or
modifies the public data files under `data/downloads/fluid-benchmarking/`. A failed run is written
with the worst value of every scored metric (rank distance 102, total variation 100, saturation 0).

`bash scripts/baseline.sh [random]` serves the paper's random-subset ablations through the same
protocol; `method/serve.py` is the reference implementation of the protocol.

## Slices

| Slice | LMs (checkpoints) | Script | Output |
|---|---|---|---|
| train (visible) | Amber-6.7B (73), OLMo1-7B (83), OLMo2-7B (94) | `scripts/evaluate_train.sh` (also `scripts/evaluate.sh`) | `scoring/scores_train.json` (`scoring/scores.json`) |
| test (hidden) | K2-65B (46), Pythia-2.8B (78), Pythia-6.9B (78) | `scripts/evaluate_test.sh` | `scoring/scores_test.json` |

Both scripts take no arguments, run the same protocol and compute the same metrics. The held-out
LMs' responses are available only to the hidden evaluator. Every checkpoint of every slice LM is
evaluated on all six benchmarks at all four budgets (train: 6,000 sessions; test: 4,848).

## Metrics

For a slice, budget and method, each LM/benchmark pair gives a score curve x_1..x_n over the
checkpoints in training order.

1. **Validity — `rank_distance` (lower is better).** For the benchmark pairs ARC Challenge ↔ MMLU
   and HellaSwag ↔ WinoGrande, in both directions: rank the checkpoint's score on the source
   benchmark among the 102 reference LMs' values on that benchmark (abilities or accuracies, per the
   declared scale; rank = 1 + number of reference LMs strictly higher), rank the checkpoint's
   full-benchmark accuracy on the target benchmark among the reference LMs' accuracies, and take the
   absolute rank difference. Averaged over directions, LMs and checkpoints. Full-benchmark
   accuracies are computed by the evaluator from all items. GSM8K and TruthfulQA do not enter
   validity.
2. **Variance — `total_variation` (lower is better).**
   TV = n/(n−1) · Σ_t |x_{t+1} − x_t| / |x_n − x_1|, capped at 100; a curve whose last value equals
   its first (including a constant curve) scores 100. Averaged over the LM × benchmark curves. A
   perfectly monotone curve scores n/(n−1) ≈ 1.
3. **Saturation — `saturation` (higher is better).** Absolute Spearman correlation between
   checkpoint index and score (average ranks for ties; a constant curve scores 0). Averaged over the
   LM × benchmark curves.
4. **Efficiency** is the budget sweep itself: the method may use at most 10/50/100/500 items.

## Experiments in the score files

| Experiment | Scored rows (`results`) | Unranked context |
|---|---|---|
| `table2_ablation_comparison` (weight 0.65) | `submitted_10`, `submitted_50`, `submitted_100`, `submitted_500` | `baselines`: RANDOM and RANDOM IRT at each budget |
| `appendix_h_full_accuracy_comparison` (0.15) | `submitted_500` | `baselines`: full-benchmark accuracy, all-item MAP ability |
| `appendix_f_table4_worst_lm_100` (0.20) | `submitted_100`: worst LM of the slice (max rank distance, max TV, min saturation) | `per_lm`: per-LM values of the method and of both baselines |

Baselines are computed by the evaluator, not the method: RANDOM = accuracy on a random item subset,
RANDOM IRT = MAP ability (2PL, N(0,1) prior, [−4, 4]) on the same subset; each subset is fixed per
benchmark and budget, shared by all checkpoints, and results are averaged over five fixed subsets.
Only the `results` rows are ranked; coefficients are −1 (rank distance), −0.8 (total variation) and
+1 (saturation) per experiment.

## Paper's numbers (all six LMs)

| Method | Items | Validity ↓ | Variance ↓ | Saturation ↑ |
|---|---:|---:|---:|---:|
| RANDOM | 10 / 50 / 100 / 500 | 20.0 / 15.2 / 16.9 / 9.1 | 29.0 / 19.1 / 19.8 / 10.2 | 0.47 / 0.62 / 0.64 / 0.79 |
| RANDOM IRT | 10 / 50 / 100 / 500 | 14.1 / 11.1 / 10.6 / 8.4 | 18.2 / 15.7 / 17.8 / 10.9 | 0.48 / 0.69 / 0.71 / 0.85 |
| Paper's proposed method | 10 / 50 / 100 / 500 | 10.1 / 8.8 / 8.7 / 8.3 | 10.7 / 6.5 / 6.1 / 4.9 | 0.76 / 0.86 / 0.85 / 0.88 |
| Full-benchmark accuracy | all | 9.1 | 23.8 | 0.85 |

Each gym slice holds three of the six LMs, so slice values differ somewhat from these averages.
