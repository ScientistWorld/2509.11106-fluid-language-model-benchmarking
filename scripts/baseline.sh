#!/bin/bash
# Run a baseline method from the paper.
#
# In this evaluation, we compare three methods:
# 1. RANDOM: random item sampling + accuracy
# 2. RANDOM IRT: random item sampling + IRT ability estimation
# 3. FLUID: adaptive item selection + IRT ability estimation
#
# Usage:
#   bash scripts/baseline.sh <name>
#   where name is 'random' or 'random_irt'

set -e

cd /home/user

BASELINE="${1:-random}"

echo "=== Running baseline: $BASELINE ==="

python3 - <<PYEOF
import os, sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/user')
sys.path.insert(0, '/home/user/method')
sys.path.insert(0, '/home/user/data')

from data.loaders import (
    load_irt_models, load_lm_evaluation_results,
    get_item_ids_for_benchmark, select_even_checkpoints, BENCHMARKS
)
from method.irt import estimate_ability
from method.fluid import FluidBenchmarking

print(f"Loading data for baseline: $BASELINE")
irt_models = load_irt_models()
lm_results = load_lm_evaluation_results()

max_checkpoints = 30
lm_checkpoints = {}
for lm_name, df in lm_results.items():
    checkpoints = list(df.columns)
    selected = select_even_checkpoints(checkpoints, max_checkpoints)
    lm_checkpoints[lm_name] = selected

seed = 42
subset_sizes = [10, 50, 100]
results = {}

for lm_name, df in lm_results.items():
    results[lm_name] = {}
    checkpoints = lm_checkpoints[lm_name]
    for cp_idx, checkpoint in enumerate(checkpoints):
        results[lm_name][checkpoint] = {}
        for bm in BENCHMARKS:
            item_ids = get_item_ids_for_benchmark(df, bm)
            if len(item_ids) == 0:
                continue
            responses = df.loc[item_ids, checkpoint].values.astype(int)
            irt_df = irt_models[bm]
            from method.irt import IRTModel
            irt = IRTModel(bm, irt_df['item_id'].values, irt_df['a'].values, irt_df['b'].values)
            a_all, b_all = irt.get_all_params()

            rng = np.random.default_rng(seed + cp_idx)
            n_use = subset_sizes[-1]
            indices = rng.choice(len(responses), size=min(n_use, len(responses)), replace=False)

            if '$BASELINE' == 'random':
                score = float(np.mean(responses[indices]))
            else:
                responses_sub = responses[indices]
                a_sub = a_all[indices]
                b_sub = b_all[indices]
                score = float(estimate_ability(responses_sub, a_sub, b_sub, theta_init=0.0))

            results[lm_name][checkpoint][bm] = score

# Save
out = Path(f'/home/user/checkpoints/baseline_{"$BASELINE"}.json')
with open(out, 'w') as f:
    json.dump({'method': '$BASELINE', 'results': results, 'checkpoints': {lm: cks for lm, cks in lm_checkpoints.items()}}, f, indent=2)
print(f"Baseline results saved to {out}")
PYEOF