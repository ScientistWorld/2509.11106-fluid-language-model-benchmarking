"""
Data loading for FLUID BENCHMARKING reproduction.

Loads:
- Pre-fitted IRT models (one per benchmark)
- OLL (Open LLM Leaderboard) results for IRT model training
- LM checkpoint evaluation results for the 6 test LMs
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Data directory
DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'downloads' / 'fluid-benchmarking'

# Benchmark names used throughout
BENCHMARKS = ['arc_challenge', 'gsm8k', 'hellaswag', 'truthfulqa_mc2', 'winogrande', 'mmlu']

# Cross-benchmark pairs for validity (from paper)
VALIDITY_PAIRS = [
    ('arc_challenge', 'mmlu'),  # knowledge/reasoning
    ('hellaswag', 'winogrande'),  # commonsense reasoning
]


def load_oll_results():
    """
    Load Open LLM Leaderboard results.

    Returns:
        dict: {benchmark: {accuracy: {model: acc}, ability: {model: ability}}}
    """
    with open(DATA_DIR / 'open_llm_leaderboard_results.json') as f:
        data = json.load(f)
    return data


def load_irt_models():
    """
    Load all pre-fitted IRT models.

    Returns:
        dict: {benchmark: pd.DataFrame with columns [item_id, a, b]}
    """
    models = {}
    for bm in BENCHMARKS:
        csv_path = DATA_DIR / 'irt_models' / f'{bm}.csv'
        df = pd.read_csv(csv_path)
        # Rename first column to item_id
        first_col = df.columns[0]
        if first_col != 'item_id':
            df = df.rename(columns={first_col: 'item_id'})
        models[bm] = df
    return models


def load_lm_evaluation_results():
    """
    Load evaluation results for all 6 test LMs across their checkpoints.

    Returns:
        dict: {lm_name: pd.DataFrame} where each df has item_id as index
              and checkpoint columns with 0/1 responses
    """
    files = {
        'amber-7b': 'amber-7b.csv',
        'k2-65b': 'k2-65b.csv',
        'olmo1-7b': 'olmo1-7b.csv',
        'olmo2-7b': 'olmo2-7b.csv',
        'pythia-3b': 'pythia-3b.csv',
        'pythia-7b': 'pythia-7b.csv',
    }

    results = {}
    for lm_name, filename in files.items():
        df = pd.read_csv(DATA_DIR / 'lm_eval_results' / filename)
        first_col = df.columns[0]
        if first_col != 'item_id':
            df = df.rename(columns={first_col: 'item_id'})
        df = df.set_index('item_id')
        results[lm_name] = df
    return results


def get_checkpoint_columns(lm_results_df):
    """
    Get all checkpoint column names from a results dataframe.

    Returns:
        list of column names (checkpoints)
    """
    return list(lm_results_df.columns)


def get_item_ids_for_benchmark(lm_results_df, benchmark):
    """
    Get item IDs for a specific benchmark from LM results.

    Args:
        lm_results_df: dataframe with item_id as index
        benchmark: benchmark name

    Returns:
        list of item IDs for this benchmark
    """
    all_items = lm_results_df.index.tolist()
    prefix = f"{benchmark}_"
    return [item for item in all_items if item.startswith(prefix) or item == benchmark]


def extract_responses(lm_results_df, benchmark, checkpoint):
    """
    Extract item responses for a specific benchmark and checkpoint.

    Args:
        lm_results_df: dataframe with item_id as index
        benchmark: benchmark name
        checkpoint: checkpoint column name

    Returns:
        tuple (item_ids, responses) where responses is array of 0/1
    """
    item_ids = get_item_ids_for_benchmark(lm_results_df, benchmark)
    responses = lm_results_df.loc[item_ids, checkpoint].values.astype(int)
    return item_ids, responses


def build_response_matrix(lm_results_df, benchmark, checkpoints):
    """
    Build a response matrix for a benchmark across checkpoints.

    Args:
        lm_results_df: dataframe with item_id as index
        benchmark: benchmark name
        checkpoints: list of checkpoint column names

    Returns:
        dict: {checkpoint: (item_ids, responses_array)}
    """
    item_ids = get_item_ids_for_benchmark(lm_results_df, benchmark)
    result = {}
    for cp in checkpoints:
        responses = lm_results_df.loc[item_ids, cp].values.astype(int)
        result[cp] = (item_ids, responses)
    return result


def get_oll_response_matrix(oll_results, benchmark):
    """
    Build response matrix from OLL results for IRT model training.

    Args:
        oll_results: loaded OLL results dict
        benchmark: benchmark name

    Returns:
        tuple (models, items, response_matrix)
        - models: list of model names
        - items: list of item IDs
        - response_matrix: np.array of shape (n_models, n_items) with 0/1
    """
    bm_data = oll_results[benchmark]
    accuracy = bm_data['accuracy']

    models = list(accuracy.keys())
    # Build item set
    if benchmark == 'mmlu':
        item_ids = [f"mmlu_{cat}_{i}" for cat in ['test', 'dev', 'val']
                    for i in range(10000)]
    else:
        item_ids = None

    # For the paper, they used item-level data from OLL which may differ
    # from the checkpoint data. Let's check.
    return models, accuracy, benchmark


def get_lm_metadata():
    """
    Get metadata about the 6 test LMs.

    Returns:
        dict: {lm_name: {family, size, n_checkpoints, checkpoint_col_prefix}}
    """
    return {
        'amber-7b': {
            'family': 'Amber',
            'size': '6.7B',
            'n_checkpoints': 73,
            'prefix': 'ckpt_',
        },
        'k2-65b': {
            'family': 'K2',
            'size': '65B',
            'n_checkpoints': 61,
            'prefix': 'ckpt_',
        },
        'olmo1-7b': {
            'family': 'OLMo1',
            'size': '7B',
            'n_checkpoints': 83,
            'prefix': 'step',
        },
        'olmo2-7b': {
            'family': 'OLMo2',
            'size': '7B',
            'n_checkpoints': 94,
            'prefix': 'step',
        },
        'pythia-3b': {
            'family': 'Pythia',
            'size': '2.8B',
            'n_checkpoints': 78,
            'prefix': 'step',
        },
        'pythia-7b': {
            'family': 'Pythia',
            'size': '6.9B',
            'n_checkpoints': 78,
            'prefix': 'step',
        },
    }


def select_even_checkpoints(checkpoints, n_desired):
    """
    Select evenly spaced checkpoints from a list.

    Args:
        checkpoints: list of checkpoint identifiers
        n_desired: number of checkpoints to select

    Returns:
        list of selected checkpoints
    """
    n_total = len(checkpoints)
    if n_total <= n_desired:
        return checkpoints

    indices = np.linspace(0, n_total - 1, n_desired).astype(int)
    return [checkpoints[i] for i in indices]


if __name__ == '__main__':
    # Test data loading
    print("Loading data...")
    oll = load_oll_results()
    print(f"OLL results: {list(oll.keys())}")

    irt_models = load_irt_models()
    print(f"IRT models: {list(irt_models.keys())}")
    for bm, df in irt_models.items():
        print(f"  {bm}: {len(df)} items")

    lm_results = load_lm_evaluation_results()
    print(f"LM results: {list(lm_results.keys())}")
    for lm, df in lm_results.items():
        print(f"  {lm}: {df.shape[0]} items x {df.shape[1]} checkpoints")
