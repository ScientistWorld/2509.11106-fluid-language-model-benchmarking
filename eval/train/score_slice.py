"""Train-slice scorer for FLUID BENCHMARKING result payloads.

This file is intentionally self-contained: eval/train must not import from
eval/test, and the scorer must not import method code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / "scoring" / "reference.json"
DEFAULT_RESULTS = REPO_ROOT / "checkpoints" / "experiment_results.json"
DEFAULT_OUTPUT = REPO_ROOT / "scoring" / "scores_train.json"

TABLE2_EXPERIMENT = "table2_ablation_comparison"
APPENDIX_H_EXPERIMENT = "appendix_h_full_accuracy_comparison"
TABLE4_EXPERIMENT = "appendix_f_table4_lm_breakdown_100"
BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
VALIDITY_PAIRS = [("arc_challenge", "mmlu"), ("hellaswag", "winogrande")]
TRAIN_LMS = ["amber-7b", "olmo1-7b", "olmo2-7b"]
TABLE4_RESULT_NAMES = {
    "amber-7b": "amber_7b",
    "k2-65b": "k2_65b",
    "olmo1-7b": "olmo1_7b",
    "olmo2-7b": "olmo2_7b",
    "pythia-3b": "pythia_3b",
    "pythia-7b": "pythia_7b",
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def compute_total_variation(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.0
    total_variation = sum(abs(scores[index + 1] - scores[index]) for index in range(n - 1))
    range_value = abs(scores[-1] - scores[0])
    if range_value < 1e-10:
        return 0.0 if total_variation < 1e-10 else 100.0
    return min((n / (n - 1)) * total_variation / range_value, 100.0)


def average_tied_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for offset in range(start, end):
            ranks[order[offset]] = average_rank
        start = end
    return ranks


def compute_saturation(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 1.0
    if len(set(scores)) <= 1:
        return 0.0

    ranks = average_tied_ranks(scores)
    steps = [float(index) for index in range(n)]
    ranks_mean = sum(ranks) / n
    steps_mean = sum(steps) / n
    numerator = sum((rank - ranks_mean) * (step - steps_mean) for rank, step in zip(ranks, steps))
    rank_norm = sum((rank - ranks_mean) ** 2 for rank in ranks)
    step_norm = sum((step - steps_mean) ** 2 for step in steps)
    denom = math.sqrt(rank_norm * step_norm)
    if denom == 0.0:
        return 0.0
    return abs(numerator / denom)


def descending_ordinal_ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0] * len(values)
    for ordinal, index in enumerate(order):
        ranks[index] = len(values) - ordinal
    return ranks


def compute_rank_distance(source_scores: list[float], target_scores: list[float]) -> float:
    n = len(source_scores)
    if n < 2:
        return 0.0
    source_ranks = descending_ordinal_ranks(source_scores)
    target_ranks = descending_ordinal_ranks(target_scores)
    return sum(abs(source_rank - target_rank) for source_rank, target_rank in zip(source_ranks, target_ranks)) / n


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


def ordered_shared_checkpoints(source_curve: dict, target_curve: dict) -> list[str]:
    return [checkpoint for checkpoint in source_curve if checkpoint in target_curve]


def train_checkpoint_subset(checkpoints: list[str]) -> list[str]:
    return [checkpoint for index, checkpoint in enumerate(checkpoints) if index % 2 == 0]


def filtered_curve_values(curve: dict, checkpoints: list[str]) -> list[float]:
    return [float(curve[checkpoint]) for checkpoint in checkpoints]


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
                checkpoints = ordered_shared_checkpoints(source_curve, target_curve)
                if len(checkpoints) < 2:
                    continue
                distances.append(
                    compute_rank_distance(
                        filtered_curve_values(source_curve, checkpoints),
                        filtered_curve_values(target_curve, checkpoints),
                    )
                )

    return mean_or_nan(distances)


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
            curve = [float(value) for value in score_curves[method][budget][lm_name][benchmark].values()]
            tv_values.append(compute_total_variation(curve))
            saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": compute_validity(payload, method, budget, lm_names),
        "total_variation": mean_or_nan(tv_values),
        "saturation": mean_or_nan(saturation_values),
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
                checkpoints = ordered_shared_checkpoints(source_curve, target_curve)
                if len(checkpoints) < 2:
                    continue
                rank_distances.append(
                    compute_rank_distance(
                        filtered_curve_values(source_curve, checkpoints),
                        filtered_curve_values(target_curve, checkpoints),
                    )
                )

    tv_values = []
    saturation_values = []
    for lm_name in lm_names:
        for benchmark in BENCHMARKS:
            curve = [float(value) for value in accuracy_curves[lm_name][benchmark].values()]
            tv_values.append(compute_total_variation(curve))
            saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": mean_or_nan(rank_distances),
        "total_variation": mean_or_nan(tv_values),
        "saturation": mean_or_nan(saturation_values),
    }


def compute_table4_metrics(payload: dict, method: str, budget: str, lm_name: str) -> dict:
    score_curves = payload["score_curves"]
    accuracy_curves = payload["accuracy_curves"]
    benchmark_scores = score_curves[method][budget][lm_name]
    lm_accuracy_curves = accuracy_curves[lm_name]

    rank_distances = []
    for benchmark_a, benchmark_b in VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            source_curve = benchmark_scores[source_benchmark]
            target_curve = lm_accuracy_curves[target_benchmark]
            checkpoints = train_checkpoint_subset(ordered_shared_checkpoints(source_curve, target_curve))
            if len(checkpoints) < 2:
                continue
            rank_distances.append(
                compute_rank_distance(
                    filtered_curve_values(source_curve, checkpoints),
                    filtered_curve_values(target_curve, checkpoints),
                )
            )

    tv_values = []
    saturation_values = []
    for benchmark in BENCHMARKS:
        full_curve = benchmark_scores[benchmark]
        checkpoints = train_checkpoint_subset(list(full_curve.keys()))
        if len(checkpoints) < 2:
            continue
        curve = filtered_curve_values(full_curve, checkpoints)
        tv_values.append(compute_total_variation(curve))
        saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": mean_or_nan(rank_distances),
        "total_variation": mean_or_nan(tv_values),
        "saturation": mean_or_nan(saturation_values),
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

    if TABLE4_EXPERIMENT in reference["experiments"]:
        table4_ref = reference["experiments"][TABLE4_EXPERIMENT]
        table4_results = {}
        for lm_name, lm_result_name in TABLE4_RESULT_NAMES.items():
            for method in ("random", "random_irt", "fluid"):
                result_name = f"{method}_{lm_result_name}_100"
                if result_name not in table4_ref["results"]:
                    continue
                table4_results[result_name] = clean_metrics(
                    compute_table4_metrics(payload, method, "100", lm_name),
                    table4_ref,
                )
        scores["experiments"][TABLE4_EXPERIMENT] = {"results": table4_results}

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
    print("Table 4 checkpoint slice: even-indexed checkpoints per LM")


if __name__ == "__main__":
    main()
