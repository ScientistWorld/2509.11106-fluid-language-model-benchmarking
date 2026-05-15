"""Held-out test-slice scorer for FLUID BENCHMARKING result payloads.

This file is intentionally self-contained: eval/test must not import from
eval/train, and the scorer must not import method code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / "scoring" / "reference.json"
DEFAULT_RESULTS = REPO_ROOT / "checkpoints" / "experiment_results.json"
DEFAULT_OUTPUT = REPO_ROOT / "scoring" / "scores_test.json"
DEFAULT_TRAIN_SCORES = REPO_ROOT / "scoring" / "scores_train.json"
OLL_RESULTS_PATH = REPO_ROOT / "data" / "downloads" / "fluid-benchmarking" / "open_llm_leaderboard_results.json"

TABLE2_EXPERIMENT = "table2_ablation_comparison"
APPENDIX_H_EXPERIMENT = "appendix_h_full_accuracy_comparison"
TABLE4_EXPERIMENT = "appendix_f_table4_lm_breakdown_100"
TABLE4_SPLIT_SEED = "81"
BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
TEST_VALIDITY_PAIRS = [("gsm8k", "mmlu"), ("arc_challenge", "truthfulqa_mc2")]
TEST_LMS = ["k2-65b", "pythia-3b", "pythia-7b"]
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


def rank_against_reference(score: float, reference_scores: list[float]) -> int:
    return 1 + sum(reference_score > score for reference_score in reference_scores)


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


def test_checkpoint_subset(checkpoints: list[str], lm_name: str) -> list[str]:
    tagged = []
    for checkpoint in checkpoints:
        digest = hashlib.sha256(f"{TABLE4_SPLIT_SEED}:{lm_name}:{checkpoint}".encode("utf-8")).hexdigest()
        tagged.append((digest, checkpoint))
    tagged.sort()
    train_set = {checkpoint for _, checkpoint in tagged[: len(tagged) // 2]}
    return [checkpoint for checkpoint in checkpoints if checkpoint not in train_set]


def filtered_curve_values(curve: dict, checkpoints: list[str]) -> list[float]:
    return [float(curve[checkpoint]) for checkpoint in checkpoints]


def compute_validity(
    payload: dict,
    oll_results: dict,
    method: str,
    budget: str,
    lm_names: list[str],
) -> float:
    score_curves = payload["score_curves"]
    accuracy_curves = payload["accuracy_curves"]
    distances = []

    for benchmark_a, benchmark_b in TEST_VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            source_reference_key = "accuracy" if method == "random" else "ability"
            source_reference = [
                float(value)
                for value in oll_results[source_benchmark][source_reference_key].values()
            ]
            target_reference = [
                float(value)
                for value in oll_results[target_benchmark]["accuracy"].values()
            ]
            for lm_name in lm_names:
                source_curve = score_curves[method][budget][lm_name][source_benchmark]
                target_curve = accuracy_curves[lm_name][target_benchmark]
                checkpoints = ordered_shared_checkpoints(source_curve, target_curve)
                if len(checkpoints) < 2:
                    continue
                for checkpoint in checkpoints:
                    source_rank = rank_against_reference(
                        float(source_curve[checkpoint]),
                        source_reference,
                    )
                    target_rank = rank_against_reference(
                        float(target_curve[checkpoint]),
                        target_reference,
                    )
                    distances.append(abs(source_rank - target_rank))

    return mean_or_nan(distances)


def compute_method_metrics(
    payload: dict,
    oll_results: dict,
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
        "rank_distance": compute_validity(payload, oll_results, method, budget, lm_names),
        "total_variation": mean_or_nan(tv_values),
        "saturation": mean_or_nan(saturation_values),
    }


def compute_full_accuracy_metrics(payload: dict, oll_results: dict, lm_names: list[str]) -> dict:
    accuracy_curves = payload["accuracy_curves"]
    rank_distances = []
    for benchmark_a, benchmark_b in TEST_VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            source_reference = [
                float(value)
                for value in oll_results[source_benchmark]["accuracy"].values()
            ]
            target_reference = [
                float(value)
                for value in oll_results[target_benchmark]["accuracy"].values()
            ]
            for lm_name in lm_names:
                source_curve = accuracy_curves[lm_name][source_benchmark]
                target_curve = accuracy_curves[lm_name][target_benchmark]
                checkpoints = ordered_shared_checkpoints(source_curve, target_curve)
                if len(checkpoints) < 2:
                    continue
                for checkpoint in checkpoints:
                    source_rank = rank_against_reference(
                        float(source_curve[checkpoint]),
                        source_reference,
                    )
                    target_rank = rank_against_reference(
                        float(target_curve[checkpoint]),
                        target_reference,
                    )
                    rank_distances.append(abs(source_rank - target_rank))

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


def compute_table4_metrics(payload: dict, oll_results: dict, method: str, budget: str, lm_name: str) -> dict:
    score_curves = payload["score_curves"]
    accuracy_curves = payload["accuracy_curves"]
    benchmark_scores = score_curves[method][budget][lm_name]
    lm_accuracy_curves = accuracy_curves[lm_name]

    rank_distances = []
    for benchmark_a, benchmark_b in TEST_VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            source_reference_key = "accuracy" if method == "random" else "ability"
            source_reference = [
                float(value)
                for value in oll_results[source_benchmark][source_reference_key].values()
            ]
            target_reference = [
                float(value)
                for value in oll_results[target_benchmark]["accuracy"].values()
            ]
            source_curve = benchmark_scores[source_benchmark]
            target_curve = lm_accuracy_curves[target_benchmark]
            checkpoints = test_checkpoint_subset(ordered_shared_checkpoints(source_curve, target_curve), lm_name)
            if len(checkpoints) < 2:
                continue
            for checkpoint in checkpoints:
                source_rank = rank_against_reference(
                    float(source_curve[checkpoint]),
                    source_reference,
                )
                target_rank = rank_against_reference(
                    float(target_curve[checkpoint]),
                    target_reference,
                )
                rank_distances.append(abs(source_rank - target_rank))

    tv_values = []
    saturation_values = []
    for benchmark in BENCHMARKS:
        full_curve = benchmark_scores[benchmark]
        checkpoints = test_checkpoint_subset(list(full_curve.keys()), lm_name)
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

    oll_results = load_json(OLL_RESULTS_PATH)
    table_ref = reference["experiments"][TABLE2_EXPERIMENT]
    table_methods = table_ref["results"]
    scores = {"experiments": {TABLE2_EXPERIMENT: {"results": {}}}}

    for budget in available_budgets(payload):
        for method in available_methods(payload):
            method_name = f"{method}_{budget}"
            if method_name not in table_methods:
                continue
            metrics = compute_method_metrics(payload, oll_results, method, budget, lm_names)
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
                    compute_full_accuracy_metrics(payload, oll_results, lm_names),
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
            for method in available_methods(payload):
                result_name = f"{method}_{lm_result_name}_100"
                if result_name not in table4_ref["results"]:
                    continue
                table4_results[result_name] = clean_metrics(
                    compute_table4_metrics(payload, oll_results, method, "100", lm_name),
                    table4_ref,
                )
        scores["experiments"][TABLE4_EXPERIMENT] = {"results": table4_results}

    return scores


def assert_same_result_keys(test_scores: dict, train_scores_path: Path) -> None:
    if not train_scores_path.exists():
        print(f"Train scores not found at {train_scores_path}; skipping key-set check")
        return

    train_scores = load_json(train_scores_path)
    train_experiments = train_scores.get("experiments", {})
    test_experiments = test_scores.get("experiments", {})
    mismatches = []

    for experiment_name, train_experiment in train_experiments.items():
        train_keys = set(train_experiment.get("results", {}))
        test_keys = set(test_experiments.get(experiment_name, {}).get("results", {}))
        if train_keys != test_keys:
            missing = sorted(train_keys - test_keys)
            extra = sorted(test_keys - train_keys)
            mismatches.append((experiment_name, missing, extra))

    extra_experiments = sorted(set(test_experiments) - set(train_experiments))
    for experiment_name in extra_experiments:
        extra = sorted(test_experiments[experiment_name].get("results", {}))
        mismatches.append((experiment_name, [], extra))

    if mismatches:
        details = []
        for experiment_name, missing, extra in mismatches:
            details.append(
                f"{experiment_name}: missing_from_test={missing}, extra_in_test={extra}"
            )
        raise ValueError(
            "Test slice result keys must match scoring/scores_train.json; "
            + "; ".join(details)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-scores", type=Path, default=DEFAULT_TRAIN_SCORES)
    args = parser.parse_args()

    payload = load_json(args.results)
    reference = load_json(args.reference)
    scores = build_scores(payload, reference, TEST_LMS)
    assert_same_result_keys(scores, args.train_scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(scores, handle, indent=2)

    print(f"Wrote {args.output}")
    print(f"Test slice LMs: {', '.join(TEST_LMS)}")
    print("Table 4 checkpoint slice: hash-balanced held-out partition per LM")


if __name__ == "__main__":
    main()
