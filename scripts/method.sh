#!/bin/bash
# Run the paper's proposed method.
#
# This script runs FLUID BENCHMARKING evaluation using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

echo "=== FLUID BENCHMARKING: Running full evaluation ==="

# Run the main experiment
python3 - <<'PYEOF'
import os, sys, json, time, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/user')
sys.path.insert(0, '/home/user/method')
sys.path.insert(0, '/home/user/data')
sys.path.insert(0, '/home/user/eval')

from data.loaders import (
    load_irt_models, load_oll_results, load_lm_evaluation_results,
    get_item_ids_for_benchmark, extract_responses,
    select_even_checkpoints, get_lm_metadata, BENCHMARKS
)
from method.irt import IRTModel, estimate_ability
from method.fluid import FluidBenchmarking
from method.metrics import compute_variance, compute_saturation, compute_validity

print("Loading data...")
irt_models = load_irt_models()
oll = load_oll_results()
lm_results = load_lm_evaluation_results()
lm_meta = get_lm_metadata()

# Methods to compare
methods = ['random', 'random_irt', 'fluid']
subset_sizes = [10, 50, 100]

# Select checkpoints (evenly spaced)
max_checkpoints = 30
lm_checkpoints = {}
for lm_name, df in lm_results.items():
    checkpoints = list(df.columns)
    selected = select_even_checkpoints(checkpoints, max_checkpoints)
    lm_checkpoints[lm_name] = selected

print(f"Running evaluation on {len(lm_checkpoints)} LMs, {max_checkpoints} checkpoints each...")

# Storage: {method: {lm: {checkpoint: {bm: score}}}}
all_scores = {m: {} for m in methods}

seed = 42

for lm_name, df in lm_results.items():
    print(f"  Processing {lm_name}...")
    checkpoints = lm_checkpoints[lm_name]
    all_scores['random'][lm_name] = {}
    all_scores['random_irt'][lm_name] = {}
    all_scores['fluid'][lm_name] = {}

    for cp_idx, checkpoint in enumerate(checkpoints):
        all_scores['random'][lm_name][checkpoint] = {}
        all_scores['random_irt'][lm_name][checkpoint] = {}
        all_scores['fluid'][lm_name][checkpoint] = {}

        for bm in BENCHMARKS:
            item_ids = get_item_ids_for_benchmark(df, bm)
            if len(item_ids) == 0:
                continue

            responses = df.loc[item_ids, checkpoint].values.astype(int)
            irt_df = irt_models[bm]
            irt = IRTModel(bm, irt_df['item_id'].values, irt_df['a'].values, irt_df['b'].values)
            a_all, b_all = irt.get_all_params()

            # RANDOM (accuracy on random subset)
            n_use = subset_sizes[-1]
            rng = np.random.default_rng(seed + cp_idx)
            indices = rng.choice(len(responses), size=min(n_use, len(responses)), replace=False)
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

    print(f"    {lm_name}: done")

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
    'lm_checkpoints': {lm: cks for lm, cks in lm_checkpoints.items()}
}
with open(out_path, 'w') as f:
    json.dump(out_data, f, indent=2)

print(f"\nResults saved to {out_path}")

# Print summary
print("\n=== SUMMARY ===")
for method in methods:
    var_vals = [v['variance'] for v in metrics_results[method].values() if 'variance' in v]
    sat_vals = [v['saturation'] for v in metrics_results[method].values() if 'saturation' in v]
    print(f"{method}: avg variance={np.mean(var_vals):.3f}, avg saturation={np.mean(sat_vals):.3f}")

print("\nDone!")
PYEOF