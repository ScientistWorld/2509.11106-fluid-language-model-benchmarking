#!/bin/bash
# Download all data and models needed for FLUID BENCHMARKING reproduction.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Estimated total download size: ~70 MB
# Estimated disk usage after extraction: ~70 MB
#
# All data comes from the HuggingFace dataset allenai/fluid-benchmarking,
# which contains pre-fitted IRT models, OLL results, and LM evaluation data.
# These artifacts are specific to this paper and are fetched into the
# workspace so a fresh clone can reproduce the results.

set -euo pipefail

cd /home/user

DATA_DIR="/home/user/data/downloads/fluid-benchmarking"
DATASET_BASE_URL="https://huggingface.co/datasets/allenai/fluid-benchmarking/resolve/main/data"

download_file() {
    local destination="$1"
    local url="$2"
    local tmp_file

    tmp_file="$(mktemp "${destination}.tmp.XXXXXX")"
    curl --fail --location --silent --show-error --retry 3 --output "$tmp_file" "$url"
    mv "$tmp_file" "$destination"
}

mkdir -p "$DATA_DIR/irt_models" "$DATA_DIR/lm_eval_results"

echo "=== Downloading FLUID BENCHMARKING data ==="
echo "Estimated total: ~70 MB"

# 1. OLL (Open LLM Leaderboard) results
OLL_FILE="$DATA_DIR/open_llm_leaderboard_results.json"
if [ ! -f "$OLL_FILE" ]; then
    echo "Downloading open_llm_leaderboard_results.json..."
    download_file "$OLL_FILE" "$DATASET_BASE_URL/open_llm_leaderboard_results.json"
    echo "  Saved to $OLL_FILE"
else
    echo "  OLL results already exist, skipping"
fi

# 2. IRT model parameters (one CSV per benchmark)
IRT_DIR="$DATA_DIR/irt_models"
for bm in arc_challenge gsm8k hellaswag truthfulqa_mc2 winogrande mmlu; do
    CSV_FILE="$IRT_DIR/${bm}.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "Downloading irt_models/${bm}.csv..."
        download_file "$CSV_FILE" "$DATASET_BASE_URL/irt_models/${bm}.csv"
    else
        echo "  ${bm}.csv already exists, skipping"
    fi
done

# 3. LM evaluation results (6 test LMs)
LM_DIR="$DATA_DIR/lm_eval_results"
for lm in amber-7b k2-65b olmo1-7b olmo2-7b pythia-3b pythia-7b; do
    CSV_FILE="$LM_DIR/${lm}.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "Downloading lm_eval_results/${lm}.csv..."
        download_file "$CSV_FILE" "$DATASET_BASE_URL/lm_eval_results/${lm}.csv"
    else
        echo "  ${lm}.csv already exists, skipping"
    fi
done

# 4. ID to item map (optional, for reference)
MAP_FILE="$DATA_DIR/id_to_item_map.json"
if [ ! -f "$MAP_FILE" ]; then
    echo "Downloading id_to_item_map.json..."
    download_file "$MAP_FILE" "$DATASET_BASE_URL/id_to_item_map.json"
else
    echo "  id_to_item_map.json already exists, skipping"
fi

echo ""
echo "=== Download complete ==="
echo "Data directory: $DATA_DIR"
du -sh "$DATA_DIR"
