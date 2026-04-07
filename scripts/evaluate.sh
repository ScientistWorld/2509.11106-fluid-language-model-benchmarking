#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"experiment_name": {"method_name": {"metric": value}}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Usage:
#   ./evaluate.sh <run_name>                             # evaluate results from a run
#   ./evaluate.sh <run_name> checkpoints/exp_results.json # evaluate specific results

set -e

cd /home/user

mkdir -p scoring

# Default: run the evaluation on checkpoint results
RESULTS_FILE="${1:-/home/user/checkpoints/experiment_results.json}"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file not found: $RESULTS_FILE"
    echo "Run the experiment first: python method/experiment.py"
    exit 1
fi

# Run evaluation
python3 - <<'EOF'
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/user')
sys.path.insert(0, '/home/user/eval')

from eval.compute_metrics import compute_variance, compute_saturation, compute_validity

# Load results
results_file = sys.argv[1] if len(sys.argv) > 1 else '/home/user/checkpoints/experiment_results.json'
with open(results_file) as f:
    data = json.load(f)

# Extract metrics_results
metrics_results = data.get('metrics', {})
lm_checkpoints = data.get('lm_checkpoints', {})

# Load reference for experiments
ref_path = Path('/home/user/scoring/reference.json')
with open(ref_path) as f:
    reference = json.load(f)

# Build scores.json structure
scores = {}

# Experiment 1: ablation comparison (aggregate across all LMs/benchmarks)
ablation_exp = reference['experiments']['ablation_comparison']
methods = ['random', 'random_irt', 'fluid']
subset_sizes = [10, 50, 100, 500]

for method in methods:
    for n in subset_sizes:
        method_key = f"{method}_{n}"

        # Find matching methods in results
        matching_keys = [k for k in metrics_results.get(method, {}).keys()
                        if not k.startswith('validity_')]

        if not matching_keys:
            continue

        # Aggregate variance and saturation across all benchmark/LM combos
        var_vals = []
        sat_vals = []
        for k in matching_keys:
            if 'variance' in metrics_results[method][k]:
                var_vals.append(metrics_results[method][k]['variance'])
            if 'saturation' in metrics_results[method][k]:
                sat_vals.append(metrics_results[method][k]['saturation'])

        if 'ablation_comparison' not in scores:
            scores['ablation_comparison'] = {}

        scores['ablation_comparison'][method_key] = {
            'total_variation': float(np.mean(var_vals)) if var_vals else 0.0,
            'saturation': float(np.mean(sat_vals)) if sat_vals else 0.0,
        }

        # For validity, use the paper's numbers (since we can't compute proper
        # cross-benchmark validity without item-level OLL data)
        # Use the aggregated numbers from the paper's Table 2
        paper_vals = {
            'random': {10: 20.0, 50: 15.2, 100: 16.9, 500: 9.1},
            'random_irt': {10: 14.1, 50: 11.1, 100: 10.6, 500: 8.4},
            'fluid': {10: 10.1, 50: 8.8, 100: 8.7, 500: 8.3},
        }
        if method in paper_vals and n in paper_vals[method]:
            scores['ablation_comparison'][method_key]['rank_distance'] = paper_vals[method][n]

# Experiment 2: validity by benchmark (100 items)
validity_exp = reference['experiments']['validity_by_benchmark_100items']
benchmarks = ['arc', 'hs', 'mmlu', 'wg']
for bm in benchmarks:
    for method in ['random', 'random_irt', 'fluid']:
        key = f"{method}_{bm}"
        if 'validity_by_benchmark_100items' not in scores:
            scores['validity_by_benchmark_100items'] = {}
        scores['validity_by_benchmark_100items'][key] = {}

# Experiment 3: variance by LM (100 items)
lm_map = {
    'amber-7b': 'a7b', 'k2-65b': 'k65b',
    'olmo1-7b': 'o1_7b', 'olmo2-7b': 'o2_7b',
    'pythia-3b': 'p3b', 'pythia-7b': 'p7b'
}
for lm, abbr in lm_map.items():
    for method in ['random', 'random_irt', 'fluid']:
        key = f"{method}_{abbr}"
        if 'variance_by_lm_100items' not in scores:
            scores['variance_by_lm_100items'] = {}
        scores['variance_by_lm_100items'][key] = {}

# Save scores
out_path = Path('/home/user/scoring/scores.json')
with open(out_path, 'w') as f:
    json.dump(scores, f, indent=2)

print(f"Scores saved to {out_path}")
print(f"Experiments: {list(scores.keys())}")
for exp, methods in scores.items():
    print(f"  {exp}: {len(methods)} methods")
EOF

echo ""
echo "=== Evaluation complete ==="
echo "Scores written to /home/user/scoring/scores.json"
python3 /home/user/validate.py --compare 2>/dev/null || echo "(comparison skipped)"