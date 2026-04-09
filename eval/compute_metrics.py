"""
Compute evaluation metrics for FLUID BENCHMARKING.

These are the four evaluation dimensions from the paper:
1. Efficiency - varies with number of items
2. Validity - rank distance between predicted and actual performance
3. Variance - total variation of training curves
4. Saturation - Spearman rank correlation of training curves

All metrics are method-agnostic — they evaluate any set of scores.
"""

import numpy as np
from scipy.stats import spearmanr


def compute_validity(estimated_scores, actual_scores):
    """
    Compute validity as rank distance between estimated and actual scores.

    Rank distance = mean absolute difference in ranks.
    Lower is better.

    Args:
        estimated_scores: list/array of estimated scores (e.g., ability estimates)
        actual_scores: list/array of actual performance values (e.g., accuracy)

    Returns:
        Mean absolute rank distance (float)
    """
    estimated_scores = np.array(estimated_scores)
    actual_scores = np.array(actual_scores)

    if len(estimated_scores) != len(actual_scores):
        raise ValueError("Estimated and actual scores must have same length")

    if len(estimated_scores) < 2:
        return 0.0

    # Compute ranks (1 = highest)
    n = len(estimated_scores)
    est_ranks = n - np.argsort(np.argsort(estimated_scores))
    actual_ranks = n - np.argsort(np.argsort(actual_scores))

    # Mean absolute rank distance
    rank_dist = np.abs(est_ranks - actual_ranks).mean()
    return float(rank_dist)


def compute_variance(scores):
    """
    Compute normalized total variation of a training curve (Equation 6 in paper).

    TV = (n/(n-1)) * sum(|x^{t+1} - x^t|) / |x^n - x^1|

    Lower is better.

    Args:
        scores: array of performance values at consecutive checkpoints

    Returns:
        Normalized total variation (float)
    """
    scores = np.array(scores)
    n = len(scores)

    if n < 2:
        return 0.0

    # Sum of absolute step-to-step changes
    total_variation = np.sum(np.abs(np.diff(scores)))

    # Total range
    range_val = abs(scores[-1] - scores[0])

    if range_val < 1e-10:
        if total_variation < 1e-10:
            return 0.0
        return float("inf")

    # Normalized total variation
    tv = (n / (n - 1)) * total_variation / range_val
    return float(tv)


def compute_saturation(scores):
    """
    Compute saturation as monotonicity (Spearman rank correlation).

    Higher is better (1 = perfectly monotonic, benchmark not saturated).

    Args:
        scores: array of performance values ordered by training progression

    Returns:
        Absolute Spearman rank correlation (float)
    """
    scores = np.array(scores)
    n = len(scores)

    if n < 2:
        return 1.0

    if len(set(scores)) <= 1:
        return 0.0

    # Spearman correlation between checkpoint index and performance
    checkpoints = np.arange(n)
    corr, _ = spearmanr(checkpoints, scores)
    return float(abs(corr))


def compute_all_metrics(method_scores, lm_checkpoints):
    """
    Compute all metrics for a method's scores across LMs and checkpoints.

    Args:
        method_scores: dict of {lm_name: {checkpoint: score}}
        lm_checkpoints: dict of {lm_name: [checkpoint_ids]}

    Returns:
        dict with 'variance' and 'saturation' aggregated over all LMs/checkpoints
    """
    var_list = []
    sat_list = []

    for lm_name, checkpoints in lm_checkpoints.items():
        if lm_name not in method_scores:
            continue

        scores = [method_scores[lm_name].get(cp, 0.0) for cp in checkpoints]
        if len(scores) >= 2:
            var_list.append(compute_variance(scores))
            sat_list.append(compute_saturation(scores))

    return {
        'variance': float(np.mean(var_list)) if var_list else 0.0,
        'saturation': float(np.mean(sat_list)) if sat_list else 0.0,
    }


def aggregate_results(all_method_scores, lm_checkpoints, benchmark_pairs=None):
    """
    Aggregate results across all methods and dimensions.

    Args:
        all_method_scores: dict of {method: {lm: {checkpoint: score}}}
        lm_checkpoints: dict of {lm: [checkpoints]}
        benchmark_pairs: list of (bm1, bm2) pairs for validity computation

    Returns:
        dict of aggregated metrics per method
    """
    results = {}

    for method, method_scores in all_method_scores.items():
        metrics = compute_all_metrics(method_scores, lm_checkpoints)
        results[method] = metrics

    return results
