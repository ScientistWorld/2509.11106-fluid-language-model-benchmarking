"""Convert experiment outputs into scoring/scores.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "scoring" / "reference.json"
DEFAULT_RESULTS_PATH = REPO_ROOT / "checkpoints" / "experiment_results.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "scoring" / "scores.json"
TABLE2_EXPERIMENT = "table2_ablation_comparison"
APPENDIX_H_EXPERIMENT = "appendix_h_full_accuracy_comparison"
TABLE4_EXPERIMENT = "appendix_f_table4_lm_breakdown_100"
BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
VALIDITY_PAIRS = [("arc_challenge", "mmlu"), ("hellaswag", "winogrande")]
OLL_RESULTS_PATH = REPO_ROOT / "data" / "downloads" / "fluid-benchmarking" / "open_llm_leaderboard_results.json"
LM_RESULT_NAMES = {
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


def rank_against_reference(score: float, reference_scores: np.ndarray) -> int:
    """Return 1-based rank of score among reference scores, with 1 = highest."""
    return int(1 + np.sum(reference_scores > score))


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
    """Spearman rank correlation against checkpoint order, without scipy."""
    x = np.asarray(scores, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    if len(set(x.tolist())) <= 1:
        return 0.0

    # Values in the released curves rarely tie, but average tied ranks for correctness.
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


def compute_full_accuracy_metrics(payload: dict) -> dict:
    """Compute Appendix H full-benchmark accuracy metrics from saved curves."""
    accuracy_curves = payload.get("accuracy_curves")
    if not isinstance(accuracy_curves, dict):
        raise ValueError("Results payload is missing 'accuracy_curves' for Appendix H")

    oll_results = load_json(OLL_RESULTS_PATH)
    rank_distances = []
    for benchmark_a, benchmark_b in VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            source_reference = np.array(
                list(oll_results[source_benchmark]["accuracy"].values()),
                dtype=float,
            )
            target_reference = np.array(
                list(oll_results[target_benchmark]["accuracy"].values()),
                dtype=float,
            )
            for benchmark_scores in accuracy_curves.values():
                source_curve = benchmark_scores[source_benchmark]
                target_curve = benchmark_scores[target_benchmark]
                for checkpoint, source_score in source_curve.items():
                    source_rank = rank_against_reference(source_score, source_reference)
                    target_rank = rank_against_reference(
                        target_curve[checkpoint],
                        target_reference,
                    )
                    rank_distances.append(abs(source_rank - target_rank))

    tv_values = []
    saturation_values = []
    for benchmark_scores in accuracy_curves.values():
        for benchmark in BENCHMARKS:
            curve = [benchmark_scores[benchmark][cp] for cp in benchmark_scores[benchmark]]
            tv_values.append(compute_total_variation(curve))
            saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": float(np.mean(rank_distances)),
        "total_variation": float(np.mean(tv_values)),
        "saturation": float(np.mean(saturation_values)),
    }


def compute_lm_breakdown_metrics(
    payload: dict,
    method: str,
    budget: int,
    lm_name: str,
) -> dict:
    """Compute the Table 4 per-LM breakdown at a fixed item budget."""
    score_curves = payload.get("score_curves")
    accuracy_curves = payload.get("accuracy_curves")
    if not isinstance(score_curves, dict) or not isinstance(accuracy_curves, dict):
        raise ValueError("Results payload is missing score_curves/accuracy_curves")

    oll_results = load_json(OLL_RESULTS_PATH)
    benchmark_scores = score_curves[method][str(budget) if str(budget) in score_curves[method] else budget][lm_name]
    lm_accuracy_curves = accuracy_curves[lm_name]

    rank_distances = []
    for benchmark_a, benchmark_b in VALIDITY_PAIRS:
        for source_benchmark, target_benchmark in (
            (benchmark_a, benchmark_b),
            (benchmark_b, benchmark_a),
        ):
            source_reference_key = "accuracy" if method == "random" else "ability"
            source_reference = np.array(
                list(oll_results[source_benchmark][source_reference_key].values()),
                dtype=float,
            )
            target_reference = np.array(
                list(oll_results[target_benchmark]["accuracy"].values()),
                dtype=float,
            )
            source_curve = benchmark_scores[source_benchmark]
            target_curve = lm_accuracy_curves[target_benchmark]
            for checkpoint, source_score in source_curve.items():
                source_rank = rank_against_reference(source_score, source_reference)
                target_rank = rank_against_reference(
                    target_curve[checkpoint],
                    target_reference,
                )
                rank_distances.append(abs(source_rank - target_rank))

    tv_values = []
    saturation_values = []
    for benchmark in BENCHMARKS:
        curve = [benchmark_scores[benchmark][cp] for cp in benchmark_scores[benchmark]]
        tv_values.append(compute_total_variation(curve))
        saturation_values.append(compute_saturation(curve))

    return {
        "rank_distance": float(np.mean(rank_distances)),
        "total_variation": float(np.mean(tv_values)),
        "saturation": float(np.mean(saturation_values)),
    }


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


def build_scores(payload: dict, reference: dict) -> dict:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Results payload is missing the computed 'summary' section")

    ref_experiment = reference["experiments"][TABLE2_EXPERIMENT]
    ref_methods = ref_experiment["results"]
    scores = {"experiments": {TABLE2_EXPERIMENT: {"results": {}}}}

    for budget, method_metrics in summary.items():
        for method, metrics in method_metrics.items():
            method_name = f"{method}_{budget}"
            if method_name not in ref_methods:
                continue
            scores["experiments"][TABLE2_EXPERIMENT]["results"][method_name] = clean_metrics(
                metrics,
                ref_experiment,
            )

    if not scores["experiments"][TABLE2_EXPERIMENT]["results"]:
        raise ValueError("No experiment results matched the reference methods")

    if APPENDIX_H_EXPERIMENT in reference["experiments"]:
        appendix_ref = reference["experiments"][APPENDIX_H_EXPERIMENT]
        scores["experiments"][APPENDIX_H_EXPERIMENT] = {
            "results": {
                "full_accuracy": clean_metrics(
                    compute_full_accuracy_metrics(payload),
                    appendix_ref,
                )
            }
        }
        if "fluid_500" in scores["experiments"][TABLE2_EXPERIMENT]["results"]:
            scores["experiments"][APPENDIX_H_EXPERIMENT]["results"]["fluid_500"] = (
                scores["experiments"][TABLE2_EXPERIMENT]["results"]["fluid_500"]
            )

    if TABLE4_EXPERIMENT in reference["experiments"]:
        table4_ref = reference["experiments"][TABLE4_EXPERIMENT]
        scores["experiments"][TABLE4_EXPERIMENT] = {"results": {}}
        for lm_name, lm_key in LM_RESULT_NAMES.items():
            for method in ("random", "random_irt", "fluid"):
                result_name = f"{method}_{lm_key}_100"
                if result_name not in table4_ref["results"]:
                    continue
                scores["experiments"][TABLE4_EXPERIMENT]["results"][result_name] = clean_metrics(
                    compute_lm_breakdown_metrics(payload, method, 100, lm_name),
                    table4_ref,
                )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    payload = load_json(args.results)
    reference = load_json(args.reference)
    scores = build_scores(payload, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(scores, handle, indent=2)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
