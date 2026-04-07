"""
Fluid Benchmarking: IRT-based adaptive evaluation for language models.

This module implements the core FLUID BENCHMARKING method from:
"Hofmann et al. - Fluid Language Model Benchmarking (2025)"
"""

from .irt import IRTModel, estimate_ability, fisher_information, logistic
from .fluid import FluidBenchmarking
from .metrics import compute_validity, compute_variance, compute_saturation

__all__ = [
    'IRTModel',
    'estimate_ability',
    'fisher_information',
    'logistic',
    'FluidBenchmarking',
    'compute_validity',
    'compute_variance',
    'compute_saturation',
]