# Research Direction

## Research Problem
This paper addresses the problem of evaluating language models efficiently and accurately.
Traditional LLM benchmarks require thousands of items and are susceptible to benchmark saturation —
as models improve, they begin to score near ceiling on existing benchmarks, making it hard to
distinguish between models. The core research question for this gym is: can a lower-budget
benchmarking strategy produce more valid, stable, and less saturated rankings than simple subset
baselines?

## Core Direction
The scientist should work within efficient language-model benchmarking. Improvements should change
how a fixed-budget evaluation strategy estimates benchmark performance or chooses which items to
evaluate, while preserving the released data, model families, benchmark set, item budgets, and
metric definitions.

## Approach Scope
The scientist may:
- Improve fixed-budget item selection
- Improve score estimation from item-level responses
- Add constraints that preserve content coverage or difficulty coverage
- Improve how checkpoint-level curves are summarized for ranking language models

## Out of Bounds
- Using external data beyond the provided reference item metadata and item responses. The scientist
  may not re-fit item metadata on additional data.
- Changing the set of benchmarks or LMs used for evaluation. The paper's benchmark set and LM
  selection define the evaluation scope.
- Simply increasing the item count to beat the reproduced scores. The goal is a better fixed-budget
  evaluation strategy, not more items.
