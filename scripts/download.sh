#!/bin/bash
# Download every artifact the FLUID BENCHMARKING gym needs (run with internet,
# before building the container or submitting jobs). Idempotent: files that
# already match their pinned sha256 are skipped.
#
# Source: Hugging Face dataset allenai/fluid-benchmarking, pinned to revision
# e0a11a3be90972ffaa12c6e63957cfd5be3a3259 (the paper's released artifacts).
#
#   data/downloads/fluid-benchmarking/   public problem data (~55 MB)
#     irt_models/<benchmark>.csv         released 2PL item parameters (a, b)
#     open_llm_leaderboard_results.json  102 reference LMs: accuracy and ability
#     lm_eval_results/{amber-7b,olmo1-7b,olmo2-7b}.csv   visible LMs' item responses
#     id_to_item_map.json                item texts (optional, for analysis)
#   eval/test/data/hidden_responses.json held-out LMs (k2-65b, pythia-3b, pythia-7b),
#                                        binarized and bit-packed (~1 MB, committed);
#                                        rebuilt here from the release if missing.
#
# The held-out LMs' raw CSVs are fetched to a temporary directory only and are
# never stored under data/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REVISION="e0a11a3be90972ffaa12c6e63957cfd5be3a3259"
BASE_URL="https://huggingface.co/datasets/allenai/fluid-benchmarking/resolve/${REVISION}/data"
DATA_DIR="$ROOT_DIR/data/downloads/fluid-benchmarking"

declare -A SHA256=(
    [open_llm_leaderboard_results.json]=b15f1484cdba75c013e9cfba54ff884420b332923303d8ff19c604d201adf77c
    [id_to_item_map.json]=5112d9b0426a62411ae5af9bf7540752f10a5fe977f0a8e636e30c461396a6f4
    [irt_models/arc_challenge.csv]=3b59923617dcaf712197c911be88c4e217a93872b57fd2ad750cfca195a20c1e
    [irt_models/gsm8k.csv]=fa36c499035b7d98a31a8024217c6421d525a41d2e71ab05a54d3d621e877695
    [irt_models/hellaswag.csv]=3028910b401f973a320062923f07c71a8f5f4d480c74eed5f82db9cdc4c54bfa
    [irt_models/mmlu.csv]=a86a139b9138102aaa804a5b81d1d5cbaed8c8f05644073795dacc2929c9cc06
    [irt_models/truthfulqa_mc2.csv]=385897f4ad000f1666846c342e102cac25480a2e7878bd041141b142ec721631
    [irt_models/winogrande.csv]=8a6491475b2cfe9fd7aaf04eb2848af0bc7e11feda61b175aa92342f6897347d
    [lm_eval_results/amber-7b.csv]=0a1cf7c5072ae3426517b15c933918cf435cc10bf794fd25d486939752502cd3
    [lm_eval_results/olmo1-7b.csv]=b32cbdb4fb173361370cd2d35545a2498ff383ef70fc65a83fd95180ceab8dfb
    [lm_eval_results/olmo2-7b.csv]=aa666e09142b34f85891ad44b572ce8c50020ac6d245d7d0e421f9e59ac4a135
    [lm_eval_results/k2-65b.csv]=4b8afb03a180d5c790ff6209dba7e2045b052aa4129f2bbddad6f03e45ddd731
    [lm_eval_results/pythia-3b.csv]=dccbfb2ddc9f281fb98434e954e96734d7b1ac3c5368ff4a8838813dabd17dfe
    [lm_eval_results/pythia-7b.csv]=7cc9befb1178009eb6f5a352dbe6b27ea6db98c751f1e729f0c63a1b1e3b1efb
)
PUBLIC_FILES=(
    open_llm_leaderboard_results.json id_to_item_map.json
    irt_models/arc_challenge.csv irt_models/gsm8k.csv irt_models/hellaswag.csv
    irt_models/mmlu.csv irt_models/truthfulqa_mc2.csv irt_models/winogrande.csv
    lm_eval_results/amber-7b.csv lm_eval_results/olmo1-7b.csv lm_eval_results/olmo2-7b.csv
)
HIDDEN_LMS=(k2-65b pythia-3b pythia-7b)
HIDDEN_TABLE="$ROOT_DIR/eval/test/data/hidden_responses.json"
HIDDEN_TABLE_SHA256=66dd76b778a05f1d0e6872c95dda33d397bdf340123525629fcc4c5755617ded

matches() { [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

fetch() {  # fetch <relative path> <destination>
    local rel="$1" destination="$2" tmp
    if matches "$destination" "${SHA256[$rel]}"; then
        echo "  ok       $rel"
        return
    fi
    mkdir -p "$(dirname "$destination")"
    tmp="$(mktemp "${destination}.tmp.XXXXXX")"
    curl --fail --location --silent --show-error --retry 3 --output "$tmp" "$BASE_URL/$rel"
    if ! matches "$tmp" "${SHA256[$rel]}"; then
        rm -f "$tmp"
        echo "sha256 mismatch for $rel" >&2
        exit 1
    fi
    mv "$tmp" "$destination"
    echo "  fetched  $rel"
}

echo "=== Public problem data -> data/downloads/fluid-benchmarking ==="
for rel in "${PUBLIC_FILES[@]}"; do
    fetch "$rel" "$DATA_DIR/$rel"
done

# Held-out LMs must never sit in data/ (older layouts stored them there).
for lm in "${HIDDEN_LMS[@]}"; do
    rm -f "$DATA_DIR/lm_eval_results/$lm.csv"
done

echo "=== Evaluator copies of the reference population ==="
for slice in train test; do
    fetch open_llm_leaderboard_results.json "$ROOT_DIR/eval/$slice/data/oll_reference.json"
done

echo "=== Held-out test responses -> eval/test/data ==="
if matches "$HIDDEN_TABLE" "$HIDDEN_TABLE_SHA256"; then
    echo "  ok       eval/test/data/hidden_responses.json"
else
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    for lm in "${HIDDEN_LMS[@]}"; do
        fetch "lm_eval_results/$lm.csv" "$TMP_DIR/$lm.csv"
    done
    python3 "$ROOT_DIR/eval/test/build_hidden_data.py" --csv-dir "$TMP_DIR" --output "$HIDDEN_TABLE"
    matches "$HIDDEN_TABLE" "$HIDDEN_TABLE_SHA256" || { echo "rebuilt hidden table hash mismatch" >&2; exit 1; }
fi

echo "=== Download complete ==="
du -sh "$DATA_DIR"
