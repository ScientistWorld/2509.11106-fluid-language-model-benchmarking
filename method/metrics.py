"""
Evaluation metrics for FLUID BENCHMARKING.

Implements the four evaluation dimensions from the paper:
1. Efficiency - measured by varying item count
2. Validity - rank distance between predicted and actual performance
3. Variance - total variation of training curves
4. Saturation - Spearman rank correlation of training curves
"""

import numpy as np
from scipy.stats import spearmanr


def compute_validity(estimated_scores, actual_scores):
    """
    Compute validity as rank distance between estimated and actual scores.

    Rank distance = mean absolute difference in ranks between estimated
    and actual performance.

    Higher estimated scores should correspond to higher actual scores.

    Args:
        estimated_scores: list/array of estimated scores (e.g., ability estimates)
        actual_scores: list/array of actual performance values (e.g., accuracy)

    Returns:
        Mean absolute rank distance
    """
    estimated_scores = np.array(estimated_scores)
    actual_scores = np.array(actual_scores)

    if len(estimated_scores) != len(actual_scores):
        raise ValueError("Estimated and actual scores must have same length")

    if len(estimated_scores) < 2:
        return 0.0

    # Compute ranks (1 = highest)
    est_ranks = len(estimated_scores) - np.argsort(np.argsort(estimated_scores))
    actual_ranks = len(actual_scores) - np.argsort(np.argsort(actual_scores))

    # Mean absolute rank distance
    rank_dist = np.abs(est_ranks - actual_ranks).mean()
    return float(rank_dist)


def compute_validity_cross_benchmark(estimated_abilities, other_accuracies, benchmark_name=None, pair_name=None):
    """
    Compute cross-benchmark validity (the main validity measure from the paper).

    This computes how well estimated abilities on one benchmark predict
    actual performance on another benchmark that targets similar capabilities.

    Pairs from the paper:
    - ARC Challenge ↔ MMLU (knowledge/reasoning)
    - HellaSwag ↔ WinoGrande (commonsense reasoning)

    Args:
        estimated_abilities: list of estimated ability values (one per model/checkpoint)
        other_accuracies: list of actual accuracy values on the other benchmark
        benchmark_name: optional name for reporting
        pair_name: optional name for the benchmark pair

    Returns:
        Mean absolute rank distance
    """
    if len(estimated_abilities) != len(other_accuracies):
        raise ValueError("Estimated abilities and other accuracies must have same length")

    if len(estimated_abilities) < 2:
        return 0.0

    return compute_validity(estimated_abilities, other_accuracies)


def compute_variance(x):
    """
    Compute normalized total variation of a training curve (Equation 6 in paper).

    TV(m_i, Q) = (n/(n-1)) * sum(|x^{t+1} - x^t|) / |x^n - x^1|

    Lower value means lower variance and more stable evaluation.

    Args:
        x: array of performance values at consecutive checkpoints

    Returns:
        Normalized total variation
    """
    x = np.array(x)
    n = len(x)

    if n < 2:
        return 0.0

    # Sum of absolute step-to-step changes
    total_variation = np.sum(np.abs(np.diff(x)))

    # Total range
    range_val = abs(x[-1] - x[0])

    if range_val < 1e-10:
        if total_variation < 1e-10:
            return 0.0
        return 100.0

    # Normalized total variation
    tv = (n / (n - 1)) * total_variation / range_val
    return min(float(tv), 100.0)


def compute_saturation(x):
    """
    Compute saturation as monotonicity (Spearman rank correlation).

    Higher value means the benchmark consistently reflects improvement
    through training — it has not yet saturated.

    Args:
        x: array of performance values at consecutive checkpoints
            (ordered by training progression)

    Returns:
        Absolute Spearman rank correlation (1 = perfectly monotonic)
    """
    x = np.array(x)
    n = len(x)

    if n < 2:
        return 1.0

    if len(set(x)) <= 1:
        return 0.0

    # Spearman correlation between checkpoint index and performance
    checkpoints = np.arange(n)
    corr, _ = spearmanr(checkpoints, x)
    return float(abs(corr))


def evaluate_all_dimensions(evaluation_results, methods, benchmarks, lm_checkpoints):
    """
    Compute all four evaluation dimensions for a set of evaluation results.

    Args:
        evaluation_results: dict with structure {method: {benchmark: {lm: {checkpoint: score}}}}
        methods: list of method names
        benchmarks: list of benchmark names
        lm_checkpoints: dict of {lm_name: list of checkpoint_ids}

    Returns:
        dict of metrics for each method/benchmark/lm combination
    """
    results = {}

    for method in methods:
        results[method] = {}
        for benchmark in benchmarks:
            results[method][benchmark] = {}
            for lm, checkpoints in lm_checkpoints.items():
                if method not in evaluation_results:
                    continue
                if benchmark not in evaluation_results[method]:
                    continue
                if lm not in evaluation_results[method][benchmark]:
                    continue

                scores = [evaluation_results[method][benchmark][lm].get(cp, 0.0) for cp in checkpoints]

                if len(scores) < 2:
                    continue

                # Compute variance
                var = compute_variance(scores)

                # Compute saturation
                sat = compute_saturation(scores)

                results[method][benchmark][lm] = {
                    'variance': var,
                    'saturation': sat,
                    'scores': scores
                }

    return results


def rank_distance_for_cross_benchmark(ability_scores_1, accuracy_2):
    """
    Compute rank distance between ability estimates on benchmark 1
    and accuracies on benchmark 2.

    This is the primary validity metric from the paper.
    """
    return compute_validity(ability_scores_1, accuracy_2)
