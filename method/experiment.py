"""
Run the FLUID BENCHMARKING reproduction on the released benchmark artifacts.

This script evaluates the paper's three ablation methods:
- RANDOM: random subset + accuracy
- RANDOM IRT: random subset + 2PL MAP ability
- FLUID: adaptive subset by Fisher information + 2PL MAP ability

The experiment uses the released item-level checkpoint results and pre-fit IRT
parameters from the paper's Hugging Face dataset. It reports the main Table 2
metrics averaged across benchmarks and LMs for item budgets 10/50/100/500.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from data.loaders import (
    BENCHMARKS,
    VALIDITY_PAIRS,
    load_irt_models,
    load_lm_evaluation_results,
    load_oll_results,
    select_even_checkpoints,
)
from method.fluid import evaluate_with_method
from method.irt import IRTModel
from method.metrics import compute_saturation, compute_variance

METHODS = ["random", "random_irt", "fluid"]
DEFAULT_BUDGETS = [10, 50, 100, 500]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "experiment_results.json"


def rank_against_reference(score: float, reference_scores: np.ndarray) -> int:
    """Return 1-based rank of score among reference scores, with 1 = highest."""
    return int(1 + np.sum(reference_scores > score))


def compute_validity_mean(
    score_curves: dict,
    accuracy_curves: dict,
    oll_results: dict,
    method: str,
    budget: int,
) -> float:
    """
    Compute the paper's external-validity proxy using released OLL distributions.

    For each checkpoint, rank the model on benchmark A using the evaluation score
    from the method under test, then compare that rank to the rank induced by the
    checkpoint's full-benchmark accuracy on the paired benchmark B. We average
    this over both benchmark directions, all LMs, and all checkpoints.
    """
    distances = []

    for bm_a, bm_b in VALIDITY_PAIRS:
        for src_bm, tgt_bm in ((bm_a, bm_b), (bm_b, bm_a)):
            src_key = "accuracy" if method == "random" else "ability"
            src_reference = np.array(
                list(oll_results[src_bm][src_key].values()),
                dtype=float,
            )
            tgt_reference = np.array(
                list(oll_results[tgt_bm]["accuracy"].values()),
                dtype=float,
            )

            for lm_name, benchmark_scores in score_curves[method][budget].items():
                src_curve = benchmark_scores[src_bm]
                tgt_acc_curve = accuracy_curves[lm_name][tgt_bm]
                for checkpoint, src_score in src_curve.items():
                    src_rank = rank_against_reference(src_score, src_reference)
                    tgt_rank = rank_against_reference(
                        tgt_acc_curve[checkpoint],
                        tgt_reference,
                    )
                    distances.append(abs(src_rank - tgt_rank))

    return float(np.mean(distances))


def aggregate_metrics(score_curves: dict, accuracy_curves: dict, oll_results: dict) -> dict:
    """Aggregate Table 2 metrics across all methods and budgets."""
    summary: dict[str, dict[str, dict[str, float]]] = {}

    for budget in sorted(score_curves["random"].keys()):
        budget_key = str(budget)
        summary[budget_key] = {}
        for method in METHODS:
            tv_values = []
            sat_values = []
            for lm_name, benchmark_scores in score_curves[method][budget].items():
                for benchmark in BENCHMARKS:
                    checkpoints = list(benchmark_scores[benchmark].keys())
                    curve = np.array(
                        [benchmark_scores[benchmark][cp] for cp in checkpoints],
                        dtype=float,
                    )
                    tv_values.append(compute_variance(curve))
                    sat_values.append(compute_saturation(curve))

            summary[budget_key][method] = {
                "rank_distance": compute_validity_mean(
                    score_curves,
                    accuracy_curves,
                    oll_results,
                    method,
                    budget,
                ),
                "total_variation": float(np.mean(tv_values)),
                "saturation": float(np.mean(sat_values)),
                "item_budget": float(budget),
            }

    return summary


def run_experiment(
    item_budgets: list[int],
    max_checkpoints_per_lm: int | None = None,
    seed: int = 0,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """Run the reproduction experiment and save raw curves plus aggregate scores."""
    irt_tables = load_irt_models()
    oll_results = load_oll_results()
    lm_results = load_lm_evaluation_results()

    irt_models = {
        benchmark: IRTModel(
            benchmark=benchmark,
            item_ids=df["item_id"].values,
            a=df["a"].values,
            b=df["b"].values,
        )
        for benchmark, df in irt_tables.items()
    }

    score_curves = {
        method: {
            budget: {} for budget in item_budgets
        }
        for method in METHODS
    }
    accuracy_curves: dict[str, dict[str, dict[str, float]]] = {}
    selected_checkpoints = {}
    benchmark_start_abilities = {
        benchmark: float(np.mean(list(oll_results[benchmark]["ability"].values())))
        for benchmark in BENCHMARKS
    }
    benchmark_samples = {}
    random.seed(seed)
    for benchmark, irt_model in irt_models.items():
        benchmark_samples[benchmark] = {}
        n_items_total = len(irt_model.item_ids)
        for budget in item_budgets:
            n_use = min(budget, n_items_total)
            benchmark_samples[benchmark][budget] = np.array(
                random.sample(range(n_items_total), n_use)
            )

    for lm_index, (lm_name, df) in enumerate(lm_results.items()):
        checkpoints = list(df.columns)
        checkpoints = select_even_checkpoints(checkpoints, max_checkpoints_per_lm) if max_checkpoints_per_lm else checkpoints
        selected_checkpoints[lm_name] = checkpoints
        accuracy_curves[lm_name] = {}

        benchmark_payload = {}
        for benchmark in BENCHMARKS:
            item_ids = irt_models[benchmark].item_ids.tolist()
            responses = df.loc[item_ids, checkpoints].values.astype(np.float32).T
            benchmark_payload[benchmark] = {
                "item_ids": item_ids,
                "responses": responses,
                "accuracy": responses.mean(axis=1),
                "irt": irt_models[benchmark],
                "theta_init": benchmark_start_abilities[benchmark],
            }
            accuracy_curves[lm_name][benchmark] = {
                checkpoint: float(acc)
                for checkpoint, acc in zip(checkpoints, benchmark_payload[benchmark]["accuracy"])
            }

        for method in METHODS:
            for budget in item_budgets:
                score_curves[method][budget][lm_name] = {
                    benchmark: {} for benchmark in BENCHMARKS
                }

        for checkpoint_index, checkpoint in enumerate(checkpoints):
            for benchmark in BENCHMARKS:
                payload = benchmark_payload[benchmark]
                responses = payload["responses"][checkpoint_index]
                item_ids = payload["item_ids"]
                irt_model = payload["irt"]
                theta_init = payload["theta_init"]

                for budget in item_budgets:
                    for method in METHODS:
                        score = evaluate_with_method(
                            irt_model=irt_model,
                            responses=responses,
                            item_ids=item_ids,
                            method=method,
                            n_items=budget,
                            seed=seed,
                            theta_init=theta_init,
                            sample_indices=benchmark_samples[benchmark][budget],
                        )
                        score_curves[method][budget][lm_name][benchmark][checkpoint] = float(score)

    summary = aggregate_metrics(score_curves, accuracy_curves, oll_results)
    payload = {
        "methods": METHODS,
        "item_budgets": item_budgets,
        "selected_checkpoints": selected_checkpoints,
        "benchmark_start_abilities": benchmark_start_abilities,
        "summary": summary,
        "score_curves": score_curves,
        "accuracy_curves": accuracy_curves,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    payload = run_experiment(
        item_budgets=args.item_budgets,
        max_checkpoints_per_lm=args.max_checkpoints,
        seed=args.seed,
        output_path=args.output,
    )

    print("Finished FLUID BENCHMARKING experiment.")
    for budget, method_scores in payload["summary"].items():
        print(f"Budget {budget}:")
        for method, metrics in method_scores.items():
            print(
                f"  {method:<10} "
                f"rank_distance={metrics['rank_distance']:.3f} "
                f"total_variation={metrics['total_variation']:.3f} "
                f"saturation={metrics['saturation']:.3f}"
            )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
