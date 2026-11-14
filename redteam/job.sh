#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p redteam/reference_seed{0,1,2}

# Independent reruns, with all 452 released checkpoints and all six benchmarks.
for seed in 0 1 2; do
    if [ ! -f "redteam/reference_seed$seed/runtime.json" ]; then
        python3 redteam/recompute.py --seed "$seed" > "redteam/reference_seed$seed/method.log" 2>&1 &
    fi
done
wait
for seed in 0 1 2; do
    test -f "redteam/reference_seed$seed/runtime.json"
done

for name in reference_seed0 reference_seed1 reference_seed2 better_denoising weaker_no_adaptivity \
    baseline_only_shortcut read_target_data_shortcut all_items_bruteforce_shortcut \
    replace_calibration_shortcut fabricated_curves_shortcut constant_shortcut short_test_shortcut \
    omit_results_shortcut unlisted_failed_method_shortcut empty_curves_shortcut singleton_shortcut \
    null_shortcut nan_shortcut score_fields_shortcut missing_payload_shortcut; do
    bash "redteam/$name/run.sh"
done
python3 redteam/analyze.py
