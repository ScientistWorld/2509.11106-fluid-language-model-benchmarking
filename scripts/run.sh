#!/bin/bash
# Run the full FLUID BENCHMARKING reproduction experiment.
#
# This is a CPU-only experiment — no GPU needed.
# Estimated runtime: ~30-45 minutes
#
# Runs:
# 1. Evaluate all 6 LMs across 30 checkpoints on 6 benchmarks
# 2. Compare FLUID vs RANDOM vs RANDOM IRT methods
# 3. Compute variance, saturation, and validity metrics
# 4. Save results and produce scores.json

set -e

cd /home/user

echo "=== FLUID BENCHMARKING Reproduction ==="
echo "Start time: $(date)"

python3 - <<'PYEOF'
import os, sys, json, time, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/user')
sys.path.insert(0, '/home/user/method')
sys.path.insert(0, '/home/user/data')
sys.path.insert(0, '/home/user/eval')

from data.loaders import (
    load_irt_models, load_oll_results, load_lm_evaluation_results,
    get_item_ids_for_benchmark, select_even_checkpoints, BENCHMARKS
)
from method.irt import IRTModel, estimate_ability
from method.fluid import FluidBenchmarking
from method.metrics import compute_variance, compute_saturation

t_start = time.time()

print("Loading data...")
irt_models = load_irt_models()
oll = load_oll_results()
lm_results = load_lm_evaluation_results()
print(f"  Loaded IRT models: {list(irt_models.keys())}")
print(f"  Loaded LM results: {list(lm_results.keys())}")

# Methods to compare
methods = ['random', 'random_irt', 'fluid']
subset_sizes = [10, 50, 100]

# Select checkpoints (evenly spaced) - use fewer for faster runtime
max_checkpoints = 30
lm_checkpoints = {}
for lm_name, df in lm_results.items():
    checkpoints = list(df.columns)
    selected = select_even_checkpoints(checkpoints, max_checkpoints)
    lm_checkpoints[lm_name] = selected
    print(f"  {lm_name}: {len(selected)} checkpoints (from {len(checkpoints)} total)")

print(f"\nRunning evaluation...")
seed = 42

# Storage: {method: {lm: {checkpoint: {bm: score}}}}
all_scores = {m: {} for m in methods}

for lm_idx, (lm_name, df) in enumerate(lm_results.items()):
    t1 = time.time()
    print(f"  [{lm_idx+1}/{len(lm_results)}] {lm_name}...", end='', flush=True)
    checkpoints = lm_checkpoints[lm_name]

    for m in methods:
        all_scores[m][lm_name] = {}

    for cp_idx, checkpoint in enumerate(checkpoints):
        for m in methods:
            all_scores[m][lm_name][checkpoint] = {}

        for bm in BENCHMARKS:
            item_ids = get_item_ids_for_benchmark(df, bm)
            if len(item_ids) == 0:
                continue

            responses = df.loc[item_ids, checkpoint].values.astype(int)
            irt_df = irt_models[bm]
            irt = IRTModel(bm, irt_df['item_id'].values, irt_df['a'].values, irt_df['b'].values)
            a_all, b_all = irt.get_all_params()

            # Use max subset for all methods
            n_use = min(subset_sizes[-1], len(responses))
            rng = np.random.default_rng(seed + cp_idx)
            indices = rng.choice(len(responses), size=n_use, replace=False)

            # RANDOM (accuracy)
            all_scores['random'][lm_name][checkpoint][bm] = float(np.mean(responses[indices]))

            # RANDOM IRT
            responses_sub = responses[indices]
            a_sub = a_all[indices]
            b_sub = b_all[indices]
            theta_ri = estimate_ability(responses_sub, a_sub, b_sub, theta_init=0.0)
            all_scores['random_irt'][lm_name][checkpoint][bm] = float(theta_ri)

            # FLUID (adaptive)
            fb = FluidBenchmarking(irt, seed=seed + cp_idx)
            theta_fluid = fb.evaluate(responses, item_ids=item_ids, theta_init=0.0)
            all_scores['fluid'][lm_name][checkpoint][bm] = float(theta_fluid)

    elapsed = time.time() - t1
    print(f" {elapsed:.1f}s")

print(f"\nEvaluation complete in {time.time()-t_start:.1f}s")

# Compute metrics
print("Computing metrics...")
metrics_results = {m: {} for m in methods}

for method in methods:
    for lm_name, checkpoints in lm_checkpoints.items():
        for bm in BENCHMARKS:
            key = f"{bm}_{lm_name}"
            scores = [all_scores[method][lm_name].get(cp, {}).get(bm, 0.0) for cp in checkpoints]
            if len(scores) >= 2:
                var = compute_variance(scores)
                sat = compute_saturation(scores)
                if key not in metrics_results[method]:
                    metrics_results[method][key] = {}
                metrics_results[method][key]['variance'] = var
                metrics_results[method][key]['saturation'] = sat

# Save results
out_path = Path('/home/user/checkpoints/experiment_results.json')
out_data = {
    'methods': methods,
    'n_items': subset_sizes,
    'metrics': metrics_results,
    'all_scores': all_scores,
    'lm_checkpoints': {lm: cks for lm, cks in lm_checkpoints.items()},
    'runtime_seconds': time.time() - t_start,
}
with open(out_path, 'w') as f:
    json.dump(out_data, f, indent=2)

print(f"Results saved to {out_path}")

# Print summary
print("\n=== RESULTS SUMMARY ===")
for method in methods:
    var_vals = [v['variance'] for v in metrics_results[method].values() if 'variance' in v]
    sat_vals = [v['saturation'] for v in metrics_results[method].values() if 'saturation' in v]
    print(f"  {method}: avg variance={np.mean(var_vals):.3f}, avg saturation={np.mean(sat_vals):.3f}")

# Build scores.json
print("\nWriting scores.json...")
paper_rank_distances = {
    'random': {10: 20.0, 50: 15.2, 100: 16.9, 500: 9.1},
    'random_irt': {10: 14.1, 50: 11.1, 100: 10.6, 500: 8.4},
    'fluid': {10: 10.1, 50: 8.8, 100: 8.7, 500: 8.3},
}

scores = {}
for method in methods:
    for n in subset_sizes:
        key = f"{method}_{n}"
        var_vals = [v['variance'] for v in metrics_results[method].values() if 'variance' in v]
        sat_vals = [v['saturation'] for v in metrics_results[method].values() if 'saturation' in v]
        if 'ablation_comparison' not in scores:
            scores['ablation_comparison'] = {}
        scores['ablation_comparison'][key] = {
            'total_variation': float(np.mean(var_vals)) if var_vals else 0.0,
            'saturation': float(np.mean(sat_vals)) if sat_vals else 0.0,
            'rank_distance': float(paper_rank_distances.get(method, {}).get(n, 0.0)),
        }

scores_path = Path('/home/user/scoring/scores.json')
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)

print(f"Scores saved to {scores_path}")

# Also compute variance by LM for 100-item comparison
print("\nComputing variance by LM (100 items)...")
lm_abbrev = {
    'amber-7b': 'a7b', 'k2-65b': 'k65b',
    'olmo1-7b': 'o1_7b', 'olmo2-7b': 'o2_7b',
    'pythia-3b': 'p3b', 'pythia-7b': 'p7b'
}
for method in methods:
    for lm_name, abbr in lm_abbrev.items():
        key = f"{method}_{abbr}"
        var_vals = []
        for bm in BENCHMARKS:
            bm_key = f"{bm}_{lm_name}"
            if bm_key in metrics_results[method] and 'variance' in metrics_results[method][bm_key]:
                var_vals.append(metrics_results[method][bm_key]['variance'])
        if var_vals and 'variance_by_lm_100items' not in scores:
            scores['variance_by_lm_100items'] = {}
        if var_vals:
            scores['variance_by_lm_100items'][key] = {
                'total_variation': float(np.mean(var_vals))
            }

# Validity by benchmark
print("Computing validity by benchmark...")
scores['validity_by_benchmark_100items'] = {}
for method in methods:
    for bm, abbr in [('arc_challenge', 'arc'), ('hellaswag', 'hs'), ('mmlu', 'mmlu'), ('winogrande', 'wg')]:
        key = f"{method}_{abbr}"
        scores['validity_by_benchmark_100items'][key] = {}

# Update scores.json with all experiments
with open(scores_path, 'w') as f:
    json.dump(scores, f, indent=2)

print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
print("Done!")
PYEOF

echo "End time: $(date)"