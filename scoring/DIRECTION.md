# Research Direction

## Research Problem
Evaluating language models efficiently and accurately during pretraining: with a budget of 10 to
500 items per benchmark and checkpoint, produce benchmark scores whose checkpoint curves are valid
(predict a related benchmark), stable (low step-to-step variation) and non-saturated (monotone
progress), on training runs the method has never seen.

## Core Direction
Improve the per-checkpoint evaluation procedure: which items to administer given the answers so
far, and how to turn the observed answers into a score.

## Approach Scope
The scientist may:
- change item selection (information criteria, exposure or content balancing, start rules,
  multi-stage designs, use of item discrimination and difficulty);
- change score estimation (estimators, priors, the score scale, calibration learned from the
  visible LMs);
- use the visible LMs' complete responses to design or tune the procedure;
- serve any procedure through `scripts/method.sh` in any language available in the container.

## Out of Bounds
- Sharing information between sessions, e.g. smoothing a checkpoint's score with other
  checkpoints' (the replay check fails such runs).
- Querying more items than the budget, or obtaining the held-out LMs' responses outside the protocol.
- Re-fitting item parameters on additional data, changing the benchmarks, LMs, budgets or metrics.
