# Research Direction

## Research Problem
This paper addresses the problem of evaluating language models efficiently and accurately.
Traditional LLM benchmarks require thousands of items and are susceptible to benchmark saturation —
as models improve, they begin to score near ceiling on existing benchmarks, making it hard to
distinguish between models. The paper's core question is: can we estimate LLM ability using far
fewer items by selecting them adaptively, and does this produce more valid, less saturated rankings?

## Core Contribution
The paper's contribution is FLUID BENCHMARKING: a method that combines Item Response Theory (IRT)
with adaptive item selection. Instead of random or fixed item sets, FLUID selects items based on
their Fisher information content — choosing items that are maximally informative for the specific
ability level being estimated. The scientist must work on improving this combination of adaptive
selection with IRT-based ability estimation.

## Approach Scope
The scientist should focus on improving the adaptive selection mechanism or the ability estimation
procedure. They may:
- Improve the item selection criterion (e.g., different information measures, uncertainty-based
  selection, diversity-aware selection)
- Improve the ability estimation procedure (e.g., different prior distributions, hierarchical Bayesian
  models, online updating)
- Add constraints on item selection (e.g., content coverage, difficulty balancing)
- Modify the IRT model (e.g., 3PL with guessing, multi-dimensional IRT)
- Improve multi-checkpoint aggregation for LM ranking

## Out of Bounds
- Switching to a non-IRT ability estimation framework entirely (e.g., purely accuracy-based or
  Elo-based rating without IRT). The paper's thesis is that IRT enables principled adaptive
  selection — dropping IRT sidesteps that contribution.
- Using external data beyond the pre-fitted IRT parameters and item responses provided by the
  paper. The scientist may not re-fit IRT models on additional data.
- Changing the set of benchmarks or LMs used for evaluation. The paper's benchmark set and LM
  selection define the evaluation scope.
- Simply increasing the item count to beat FLUID — the goal is better selection, not more items.
