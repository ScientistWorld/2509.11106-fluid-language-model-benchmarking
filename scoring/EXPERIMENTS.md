# Experiment Overview

All three experiments score the same thing: the submitted benchmarking method (`scripts/method.sh`),
run by the evaluator through the item-query protocol of `briefing/evaluation.md` on every checkpoint
of the slice's language models, all six benchmarks and all four item budgets. Only the rows under
each experiment's `results` are ranked. Baselines, ground truth, reference population, budgets and
the checkpoint roster are fixed by the evaluator.

## table2_ablation_comparison (weight 0.65)
Whether a fixed-budget benchmarking method yields valid, stable and non-saturated checkpoint
curves. Scored rows `submitted_10`, `submitted_50`, `submitted_100`, `submitted_500` report
validity (`rank_distance`), variance (`total_variation`) and saturation (`saturation`) at each
item budget, averaged over the slice's LMs, benchmarks and checkpoints.

Unranked `baselines` at the same budgets, computed by the evaluator on five fixed random item
subsets per benchmark and budget:
- `random_<budget>`: accuracy on a random item subset.
- `random_irt_<budget>`: maximum-a-posteriori 2PL ability on the same random subset.

## appendix_h_full_accuracy_comparison (weight 0.15)
Whether a 500-item evaluation can beat evaluating the whole benchmark. Scored row `submitted_500`
(the submitted method's 500-item result). Unranked `baselines`: `full_accuracy` (accuracy over all
items of each benchmark) and `full_ability` (MAP ability over all items).

## appendix_f_table4_worst_lm_100 (weight 0.20)
Whether the quality holds for every training trajectory rather than only on average. Scored row
`submitted_100`: at the 100-item budget, the worst LM of the slice on each metric (largest per-LM
rank distance, largest per-LM total variation, smallest per-LM saturation). Unranked `baselines`
are the same worst-LM values for RANDOM and RANDOM IRT; `per_lm` lists every LM's values for the
submitted method and both baselines.
