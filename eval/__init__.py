"""
Method-agnostic evaluation package for the FLUID benchmarking gym.

The active evaluator is run as ``python -m eval.score_results``. Keep this
package initializer lightweight so module execution does not require optional
metric-helper dependencies before they are actually used.
"""

__all__ = [
    "aggregate_results",
    "compute_all_metrics",
    "compute_saturation",
    "compute_validity",
    "compute_variance",
    "score_and_save",
]


def __getattr__(name):
    if name in {
        "compute_validity",
        "compute_variance",
        "compute_saturation",
        "compute_all_metrics",
        "aggregate_results",
    }:
        from . import compute_metrics

        return getattr(compute_metrics, name)
    if name == "score_and_save":
        from .scoring import score_and_save

        return score_and_save
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
