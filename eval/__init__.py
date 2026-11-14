"""Evaluators for the FLUID benchmarking gym.

``eval/train/score.py`` (visible LMs) and ``eval/test/score.py`` (held-out LMs)
are independent, self-contained scripts; they share no code with each other and
never import ``method/``. Run them through ``scripts/evaluate_train.sh`` and
``scripts/evaluate_test.sh``.
"""
