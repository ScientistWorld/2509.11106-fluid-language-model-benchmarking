"""Train-slice scorer for FLUID BENCHMARKING result payloads.

This file is intentionally self-contained: eval/train must not import from
eval/test, and the scorer must not import method code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / "scoring" / "reference.json"
DEFAULT_RESULTS = REPO_ROOT / "checkpoints" / "experiment_results.json"
DEFAULT_OUTPUT = REPO_ROOT / "scoring" / "scores_train.json"

TABLE2_EXPERIMENT = "table2_ablation_comparison"
APPENDIX_H_EXPERIMENT = "appendix_h_full_accuracy_comparison"
BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
VALIDITY_PAIRS = [("arc_challenge", "mmlu"), ("hellaswag", "winogrande")]
TRAIN_LMS = ["amber-7b", "olmo1-7b", "olmo2-7b"]


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def compute_total_variation(scores: list[float]) -> float:
    x = np.asarray(scores, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    total_variation = float(np.sum(np.abs(np.diff(x))))
    range_value = float(abs(x[-1] - x[0]))
    if range_value < 1e-10:
        return 0.0 if total_variation < 1e-10 else 100.0
    return min(float((n / (n - 1)) * total_variation / range_value), 100.0)


def compute_saturation(scores: list[float]) -> float:
    x = np.asarray(scores, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    if len(set(x.tolist())) <= 1:
        return 0.0

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end

    steps = np.arange(n, dtype=float)
    ranks -= ranks.mean()
    steps -= steps.mean()
    denom = np.sqrt(np.sum(ranks * ranks) * np.sum(steps * steps))
    if denom == 0.0:
        return 0.0
    return float(abs(np.sum(ranks * steps) / denom))


def compute_rank_distance(source_scores: list[float], target_scores: list[float]) -> float:
    source = np.asarray(source_scores, dtype=float)
    target = np.asarray(target_scores, dtype=float)
    n = len(source)
    if n < 2:
        return 0.0

    source_ranks = n - np.argsort(np.argsort(source, kind="mergesort"), kind="mergesort")
    target_ranks = n - np.argsort(np.argsort(target, kind="mergesort"), kind="mergesort")
    return float(np.mean(np.abs(source_ranks - target_ranks)))


def available_budgets(payload: dict) -> list[str]:
    if "item_budgets" in payload:
        return [str(budget) for budget in payload["item_budgets"]]
    return sorted(payload["score_curves"]["fluid"].keys(), key=int)


def available_methods(payload: dict) -> list[str]:
    return list(payload.get("methods") or payload["score_curves"].keys())


def clean_metrics(metrics: dict, ref_experiment: dict) -> dict:
    cleaned = {}
    for metric_name in ref_experiment["metrics"]:
        if metric_name not in metrics:
            continue
        value = float(metrics[metric_name])
        if not math.isfinite(value):
            value = 100.0
        cleaned[metric_name] = value
    return {
        metric_name: cleaned[metric_name]
        for metric_name in ref_experiment["metrics"]
        if metric_name in cleaned
    }


def compute_validity(
    payload: dict,
    method: str,
    budget: str,
    lm_names: list[str],
) -> float:
    score_curves = payload["score_curves"]
    accuracy_curves = payload["accuracy_curves"]
    distances = []

    for benchmark_a, benchmark_b in VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            for lm_name in lm_names:
                source_curve = score_curves[method][budget][lm_name][source_benchmark]
                target_curve = accuracy_curves[lm_name][target_benchmark]
                checkpoints = [checkpoint for checkpoint in source_curve if checkpoint in target_curve]
                if len(checkpoints) < 2:
                    continue
                distances.append(
                    compute_rank_distance(
                        [source_curve[checkpoint] for checkpoint in checkpoints],
                        [target_curve[checkpoint] for checkpoint in checkpoints],
                    )
                )

    return float(np.mean(distances))


def compute_method_metrics(
    payload: dict,
    method: str,
    budget: str,
    lm_names: list[str],
) -> dict:
    score_curves = payload["score_curves"]
    tv_values = []
    saturation_values = []
    for lm_name in lm_names:
        for benchmark in BENCHMARKS:
            curve = list(score_curves[method][budget][lm_name][benchmark].values())
            tv_values.append(compute_total_variation(curve))
            saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": compute_validity(payload, method, budget, lm_names),
        "total_variation": float(np.mean(tv_values)),
        "saturation": float(np.mean(saturation_values)),
    }


def compute_full_accuracy_metrics(payload: dict, lm_names: list[str]) -> dict:
    accuracy_curves = payload["accuracy_curves"]
    rank_distances = []
    for benchmark_a, benchmark_b in VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            for lm_name in lm_names:
                source_curve = accuracy_curves[lm_name][source_benchmark]
                target_curve = accuracy_curves[lm_name][target_benchmark]
                checkpoints = [checkpoint for checkpoint in source_curve if checkpoint in target_curve]
                if len(checkpoints) < 2:
                    continue
                rank_distances.append(
                    compute_rank_distance(
                        [source_curve[checkpoint] for checkpoint in checkpoints],
                        [target_curve[checkpoint] for checkpoint in checkpoints],
                    )
                )

    tv_values = []
    saturation_values = []
    for lm_name in lm_names:
        for benchmark in BENCHMARKS:
            curve = list(accuracy_curves[lm_name][benchmark].values())
            tv_values.append(compute_total_variation(curve))
            saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": float(np.mean(rank_distances)),
        "total_variation": float(np.mean(tv_values)),
        "saturation": float(np.mean(saturation_values)),
    }


def build_scores(payload: dict, reference: dict, lm_names: list[str]) -> dict:
    if "score_curves" not in payload or "accuracy_curves" not in payload:
        raise ValueError("Results payload must contain score_curves and accuracy_curves")

    missing = sorted(set(lm_names) - set(payload["accuracy_curves"]))
    if missing:
        raise ValueError(f"Results payload is missing slice LMs: {missing}")

    table_ref = reference["experiments"][TABLE2_EXPERIMENT]
    table_methods = table_ref["results"]
    scores = {"experiments": {TABLE2_EXPERIMENT: {"results": {}}}}

    for budget in available_budgets(payload):
        for method in available_methods(payload):
            method_name = f"{method}_{budget}"
            if method_name not in table_methods:
                continue
            metrics = compute_method_metrics(payload, method, budget, lm_names)
            scores["experiments"][TABLE2_EXPERIMENT]["results"][method_name] = clean_metrics(
                metrics,
                table_ref,
            )

    if not scores["experiments"][TABLE2_EXPERIMENT]["results"]:
        raise ValueError("No slice results matched the reference methods")

    if APPENDIX_H_EXPERIMENT in reference["experiments"]:
        appendix_ref = reference["experiments"][APPENDIX_H_EXPERIMENT]
        scores["experiments"][APPENDIX_H_EXPERIMENT] = {
            "results": {
                "full_accuracy": clean_metrics(
                    compute_full_accuracy_metrics(payload, lm_names),
                    appendix_ref,
                )
            }
        }
        fluid_500 = scores["experiments"][TABLE2_EXPERIMENT]["results"].get("fluid_500")
        if fluid_500 is not None:
            scores["experiments"][APPENDIX_H_EXPERIMENT]["results"]["fluid_500"] = fluid_500

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = load_json(args.results)
    reference = load_json(args.reference)
    scores = build_scores(payload, reference, TRAIN_LMS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(scores, handle, indent=2)

    print(f"Wrote {args.output}")
    print(f"Train slice LMs: {', '.join(TRAIN_LMS)}")


if __name__ == "__main__":
    main()
