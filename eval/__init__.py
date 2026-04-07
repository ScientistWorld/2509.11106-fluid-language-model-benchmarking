"""
Evaluation module for FLUID BENCHMARKING.

This module is method-agnostic — it evaluates how well results fulfill
the paper's research goals (efficiency, validity, variance, saturation)
without knowing which method produced the results.

Called by scripts/evaluate.sh to score any method's output.
"""

from .compute_metrics import (
    compute_validity,
    compute_variance,
    compute_saturation,
    compute_all_metrics,
    aggregate_results,
)
from .scoring import score_and_save

__all__ = [
    'compute_validity',
    'compute_variance',
    'compute_saturation',
    'compute_all_metrics',
    'aggregate_results',
    'score_and_save',
]