# Design Notes — FLUID BENCHMARKING Train/Test Evaluation

## Design Assessment

The existing train/test split is fundamentally sound. Split by LM family:
- **Train slice**: amber-7b, olmo1-7b, olmo2-7b (6B-class OLMo family + Amber)
- **Test slice**: k2-65b (65B LLM360), pythia-3b, pythia-7b (Pythia family, different scale/training)

This tests whether a method generalizes to unseen model families and scales, which is the core scientific claim of FLUID: adaptive item selection works across LM types.

## Checks Completed

### Scripts — Correct
- `evaluate_train.sh` invokes `eval.train.score_slice`, writes to `scores_train.json`
- `evaluate_test.sh` invokes `eval.test.score_slice`, writes to `scores_test.json`
- No hardcoded method defaults (methods enumerated from payload)
- Both syntax-check cleanly

### Output Paths — Correct
- Train slice: `DEFAULT_OUTPUT = REPO_ROOT / "scoring" / "scores_train.json"`
- Test slice: `DEFAULT_OUTPUT = REPO_ROOT / "scoring" / "scores_test.json"`
- Scripts pass `--output` flag explicitly

### Leakage — Clean
- No cross-imports between `eval/train/` and `eval/test/`
- Both modules are self-contained
- Only comment-level mentions of each other (docstrings)

### Paper-Agreement Metrics — None
- All three metrics measure scientific quantities: rank distance (validity), total variation (variance), Spearman correlation (saturation)
- No `author`, `paper`, `published`, `vs_paper`, `deviation_from_paper`, `reproduction_error`, `fit_to_published` in metric names or formulas
- Theoretical floor is not "match the paper" — metrics are unbounded in both directions

### Score Files — Correct
- `scores_train.json` and `scores_test.json` do NOT exist (deleted by Stage 2)

### Coverage
- `table2_ablation_comparison`: Fully covered by both slices
- `appendix_h_full_accuracy_comparison`: Fully covered by both slices
- `appendix_f_table4_lm_breakdown_100`: **NOT covered** — the per-LM breakdown experiment is not handled by either train or test scorer. This experiment has 18 result rows (6 LMs × 3 methods), which would require per-LM split handling. Not adding this now to keep the design simple — the primary and secondary experiments are already split.

## Runtime Concerns for Next Iteration

1. **Metric saturation**: saturation values are Spearman correlations (0-1). In the current reproduction, saturation ranges from ~0.45 to ~0.88, so there's room in both directions. No saturation risk detected.

2. **Score computation**: The train/test evaluators read from the same payload (`experiment_results.json`) and just filter by LM. No re-execution of the method is needed — the evaluator is lightweight.

3. **Method enumeration**: Both slices use `available_methods(payload)` which reads from `payload["score_curves"].keys()`. This works as long as future agents produce score curves in the same format. No hardcoded method names.

## Improvements Made

None required — the design was solid from Stage 2. The split correctly tests generalization to unseen model families, the metrics measure scientific quantities (not paper agreement), and the scripts are clean and independent.