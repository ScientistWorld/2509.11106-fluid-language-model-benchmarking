#!/bin/bash
# Install one variant/shortcut, finish its method process, then invoke the
# original hidden test with NO ARGUMENTS. No method process survives that call.
set -euo pipefail
cd /home/user
name="$1"
folder="redteam/$name"
mkdir -p "$folder"
export PYTHONDONTWRITEBYTECODE=1

# Each case starts with the same public-data calibration and output schema.
cp redteam/_initial/data/downloads/fluid-benchmarking/open_llm_leaderboard_results.json data/downloads/fluid-benchmarking/open_llm_leaderboard_results.json
cp redteam/reference_seed0/payload.json checkpoints/experiment_results.json
bash scripts/evaluate_train.sh > "$folder/reference_train.log" 2>&1
rm -f scoring/scores_test.json
python3 redteam/prepare.py "$name"

# Omission is performed through the visible, unmodified train interface.
# Other cases also run train to leave the same state as an actual attempt.
set +e
bash scripts/evaluate_train.sh > "$folder/train.log" 2>&1
train_status=$?
[ ! -f scoring/scores_train.json ] || cp scoring/scores_train.json "$folder/scores_train.json"
bash scripts/evaluate_test.sh > "$folder/test.log" 2>&1
test_status=$?
set -e
if [ -f scoring/scores_test.json ]; then
    cp scoring/scores_test.json "$folder/scores_test.json"
fi
python3 - "$name" "$train_status" "$test_status" <<'PY'
import json, sys
from pathlib import Path
name, train_status, test_status = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Path('redteam', name, 'status.json').write_text(json.dumps({
    'name': name, 'train_exit': train_status, 'test_exit': test_status,
    'scores_file_may_be_stale': test_status != 0,
    'test_command': 'bash scripts/evaluate_test.sh',
    'process_model': 'method exits before evaluator starts; no environment or process tampering'
}, indent=2))
print(name, 'train_exit=', train_status, 'test_exit=', test_status, flush=True)
PY

# Restore only files touched by the run, including ignored checkpoint/data.
cp redteam/_initial/checkpoints/experiment_results.json checkpoints/experiment_results.json
cp redteam/_initial/data/downloads/fluid-benchmarking/open_llm_leaderboard_results.json data/downloads/fluid-benchmarking/open_llm_leaderboard_results.json
cp redteam/_initial/scoring/scores_train.json scoring/scores_train.json
cp redteam/_initial/scoring/scores_test.json scoring/scores_test.json
