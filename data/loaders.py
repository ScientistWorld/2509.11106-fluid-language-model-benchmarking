"""
Data loading for the visible part of the FLUID BENCHMARKING gym.

Loads (from data/downloads/fluid-benchmarking, fetched by scripts/download.sh):
- released 2PL item parameters (one table per benchmark; row order = item index
  used by the evaluation protocol)
- Open LLM Leaderboard reference results (accuracy and ability of 102 reference LMs)
- item-level checkpoint responses of the three visible LMs (Amber-6.7B,
  OLMo1-7B, OLMo2-7B), binarized like the paper's released loader (>= 0.5)

The three held-out LMs (K2-65B, Pythia-2.8B, Pythia-6.9B) are not available here.
"""

import csv
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'downloads' / 'fluid-benchmarking'

BENCHMARKS = ['arc_challenge', 'gsm8k', 'hellaswag', 'truthfulqa_mc2', 'winogrande', 'mmlu']
VALIDITY_PAIRS = [('arc_challenge', 'mmlu'), ('hellaswag', 'winogrande')]
VISIBLE_LMS = ['amber-7b', 'olmo1-7b', 'olmo2-7b']
ITEM_BUDGETS = [10, 50, 100, 500]


def load_oll_results():
    """{benchmark: {"accuracy": {model: acc}, "ability": {model: ability}}} for 102 reference LMs."""
    with open(DATA_DIR / 'open_llm_leaderboard_results.json') as handle:
        return json.load(handle)


def load_item_parameters():
    """{benchmark: {"ids": [...], "a": array, "b": array}} in protocol item-index order."""
    params = {}
    for benchmark in BENCHMARKS:
        with open(DATA_DIR / 'irt_models' / f'{benchmark}.csv') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            a_col, b_col = header.index('a'), header.index('b')
            rows = list(reader)
        params[benchmark] = {
            'ids': [row[0] for row in rows],
            'a': np.array([float(row[a_col]) for row in rows]),
            'b': np.array([float(row[b_col]) for row in rows]),
        }
    return params


def load_visible_responses(lm_name, item_params=None):
    """Binary responses of one visible LM.

    Returns (checkpoints, {benchmark: uint8 array [n_checkpoints, n_items]}),
    checkpoints in training order, items in protocol index order.
    """
    if lm_name not in VISIBLE_LMS:
        raise ValueError(f'{lm_name} is not a visible LM; choose from {VISIBLE_LMS}')
    item_params = item_params or load_item_parameters()
    with open(DATA_DIR / 'lm_eval_results' / f'{lm_name}.csv') as handle:
        reader = csv.reader(handle)
        checkpoints = next(reader)[1:]
        table = {row[0]: row[1:] for row in reader}
    responses = {}
    for benchmark in BENCHMARKS:
        values = np.array([[float(v) for v in table[item_id]] for item_id in item_params[benchmark]['ids']])
        responses[benchmark] = (values.T >= 0.5).astype(np.uint8)
    return checkpoints, responses


if __name__ == '__main__':
    params = load_item_parameters()
    for benchmark, entry in params.items():
        print(f'{benchmark}: {len(entry["ids"])} items')
    for lm in VISIBLE_LMS:
        checkpoints, responses = load_visible_responses(lm, params)
        print(f'{lm}: {len(checkpoints)} checkpoints')
