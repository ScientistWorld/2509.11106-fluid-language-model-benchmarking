# Scientist Constraints

## Interface
The submission is the executable `scripts/method.sh`, run by the evaluators through the line
protocol in `briefing/evaluation.md`. The method must:
- answer every session (every checkpoint × benchmark × budget) with one finite score on the scale it
  declared (`ability`, or `accuracy` in [0, 1]);
- query at most `budget` distinct items per session; the evaluator counts and enforces this;
- be a deterministic, stateless per-session procedure: a session's queries and score may depend
  only on the benchmark, the budget and the answers received in that session. No information may
  be carried between sessions (in memory, on disk or otherwise); the evaluator replays sessions in a
  fresh process and fails the run on any difference;
- not read anything under `eval/` and not modify `data/downloads/` (the evaluator fails the run if
  the public data files change).

**Why**: the paper evaluates each checkpoint independently with a fixed item budget; variance and
saturation measure the noise of that per-checkpoint evaluation. Pooling or smoothing across
checkpoints would change the measured quantity, and extra items would remove the efficiency
constraint.

## Data
Use only what this environment provides: the released item parameters, the reference LMs'
accuracies and abilities, and the visible LMs' item-level responses (for development). Do not
re-fit item parameters on additional data, and do not obtain the held-out LMs' responses
(K2-65B, Pythia-2.8B, Pythia-6.9B) from any source; the method may learn only from the visible LMs.

**Why**: the held-out runs are the test; any access to their answers beyond the protocol's queries
invalidates it.

## Item budgets and metrics
Budgets are 10, 50, 100 and 500 items. The metrics (validity rank distance against the 102
reference LMs, normalized total variation, absolute Spearman monotonicity), the benchmark pairs and
the six benchmarks are fixed by the evaluator and may not be changed.

## Compliance Checklist
- [ ] `scripts/method.sh` speaks the protocol and writes logs to stderr only
- [ ] At most `budget` queries per session; every session answered
- [ ] Deterministic and stateless across sessions; no reads of `eval/`, no writes to `data/downloads/`
- [ ] No held-out LM data, no re-fitted item parameters
