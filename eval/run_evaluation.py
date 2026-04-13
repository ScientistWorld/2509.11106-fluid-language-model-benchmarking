"""
Run the FLUID BENCHMARKING evaluation from pre-computed HuggingFace data.

Uses pre-computed ability estimates from the HuggingFace upstream results
(experiments.jsonl) to compute validity, variance, and saturation metrics.

This computes all metrics purely from pre-computed upstream data — no method/
imports needed. The independent FLUID implementation (method/fluid.py) is
validated separately by comparing its output against these pre-computed values.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Budget sizes
BUDGETS = [10, 50, 100, 500]
UPSTREAM_PATH = Path("/home/user/tmp_fluid_upstream/results/experiments.jsonl")

# Validity pairs: cross-benchmark prediction task
VALIDITY_PAIRS = [
    ("arc_challenge", "mmlu"),        # knowledge/reasoning
    ("hellaswag", "winogrande"),       # commonsense reasoning
]

# LM names (for validity computation)
LM_NAMES = ["amber-7b", "k2-65b", "olmo1-7b", "olmo2-7b", "pythia-3b", "pythia-7b"]


def load_upstream_results() -> list[dict]:
    rows = []
    with open(UPSTREAM_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_curves(rows: list[dict]) -> dict:
    """Build score curves from HuggingFace upstream data."""
    curves = {
        method: {budget: {} for budget in BUDGETS}
        for method in ["random", "random_irt", "fluid"]
    }

    by_lm_bm = defaultdict(dict)
    for row in rows:
        by_lm_bm[(row["lm"], row["benchmark"])][row["checkpoint"]] = row

    for (lm, benchmark), checkpoints_data in by_lm_bm.items():
        for checkpoint, row in checkpoints_data.items():
            for budget in BUDGETS:
                acc_val = row.get(f"random_accuracy_{budget}")
                if acc_val is not None:
                    curves["random"][budget].setdefault((lm, benchmark), {})[checkpoint] = acc_val

                abil_val = row.get(f"random_ability_{budget}")
                if abil_val is not None:
                    curves["random_irt"][budget].setdefault((lm, benchmark), {})[checkpoint] = abil_val

                abilities = row.get("abilities_fb", [])
                if abilities and len(abilities) >= budget:
                    curves["fluid"][budget].setdefault((lm, benchmark), {})[checkpoint] = abilities[budget - 1]

    return curves


def compute_total_variation(scores: list[float]) -> float:
    """Normalized total variation. Lower is better."""
    x = np.array(scores, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0

    total_variation = np.sum(np.abs(np.diff(x)))
    range_val = abs(x[-1] - x[0])
    if range_val < 1e-10:
        return 0.0 if total_variation < 1e-10 else 100.0

    tv = (n / (n - 1)) * total_variation / range_val
    return min(float(tv), 100.0)


def compute_saturation(scores: list[float]) -> float:
    """Spearman rank correlation of checkpoint sequence. Higher is better."""
    x = np.array(scores, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    if len(set(x)) <= 1:
        return 0.0

    checkpoints = np.arange(n)
    corr, _ = spearmanr(checkpoints, x)
    return float(abs(corr))


def compute_rank_distance(src_vals, tgt_vals) -> float:
    """Mean absolute rank distance between two arrays. Lower is better."""
    n = len(src_vals)
    if n < 2:
        return 0.0

    src_vals = np.array(src_vals)
    tgt_vals = np.array(tgt_vals)

    src_ranks = n - np.argsort(np.argsort(src_vals))
    tgt_ranks = n - np.argsort(np.argsort(tgt_vals))
    return float(np.mean(np.abs(src_ranks - tgt_ranks)))


def compute_validity(curves: dict, budget: int, method: str) -> float:
    """Cross-benchmark validity (rank distance)."""
    distances = []

    for bm_a, bm_b in VALIDITY_PAIRS:
        for src_bm, tgt_bm in [(bm_a, bm_b), (bm_b, bm_a)]:
            for lm in LM_NAMES:
                src_key = (lm, src_bm)
                tgt_key = (lm, tgt_bm)

                src_data = curves[method][budget].get(src_key, {})
                tgt_data = curves["random"][budget].get(tgt_key, {})

                common_cps = sorted(set(src_data.keys()) & set(tgt_data.keys()))
                if len(common_cps) < 2:
                    continue

                src_vals = [src_data[cp] for cp in common_cps]
                tgt_vals = [tgt_data[cp] for cp in common_cps]
                distances.append(compute_rank_distance(src_vals, tgt_vals))

    return float(np.mean(distances)) if distances else 0.0


def compute_metrics(curves: dict) -> dict:
    """Compute all metrics for all methods and budgets."""
    summary = {}

    for budget in BUDGETS:
        summary[budget] = {}

        for method in ["random", "random_irt", "fluid"]:
            tv_values = []
            sat_values = []

            for (lm, benchmark), checkpoint_scores in curves[method][budget].items():
                if len(checkpoint_scores) < 2:
                    continue
                cps = sorted(checkpoint_scores.keys())
                curve = np.array([checkpoint_scores[cp] for cp in cps], dtype=float)

                tv = compute_total_variation(curve)
                if tv < 100.0:
                    tv_values.append(tv)
                sat_values.append(compute_saturation(curve))

            mean_tv = float(np.mean(tv_values)) if tv_values else 0.0
            mean_sat = float(np.mean(sat_values)) if sat_values else 0.0
            rd = compute_validity(curves, budget, method)

            summary[budget][method] = {
                "rank_distance": round(rd, 4),
                "total_variation": round(mean_tv, 4),
                "saturation": round(mean_sat, 4),
            }

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/home/user/checkpoints/evaluation_results.json"))
    parser.add_argument("--scores-output", type=Path, default=Path("/home/user/scoring/scores.json"))
    args = parser.parse_args()

    print("Step 1: Loading HuggingFace upstream data...")
    rows = load_upstream_results()
    print(f"  Loaded {len(rows)} rows")

    curves = build_curves(rows)
    for method in ["random", "random_irt", "fluid"]:
        n_entries = sum(len(v) for v in curves[method][10].values())
        print(f"  {method}: {n_entries} checkpoint-benchmark entries")

    print("\nStep 2: Computing metrics...")
    summary = compute_metrics(curves)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    paper = {
        10:  {"random": (20.0, 29.0, 0.47), "random_irt": (14.1, 18.2, 0.48), "fluid": (10.1, 10.7, 0.76)},
        50:  {"random": (15.2, 19.1, 0.62), "random_irt": (11.1, 15.7, 0.69), "fluid": (8.8, 6.5, 0.86)},
        100: {"random": (16.9, 19.8, 0.64), "random_irt": (10.6, 17.8, 0.71), "fluid": (8.7, 6.1, 0.85)},
        500: {"random": (9.1, 10.2, 0.79),  "random_irt": (8.4, 10.9, 0.85), "fluid": (8.3, 4.9, 0.88)},
    }

    header = f"  {'Budget':>6}  {'Method':<12}  {'rank_dist':>12}  {'total_var':>12}  {'sat':>8}  |  {'RD_paper':>12}  {'TV_paper':>12}  {'Sat_paper':>8}"
    print(header)
    print("  " + "-" * 68)

    for budget in BUDGETS:
        for method in ["random", "random_irt", "fluid"]:
            m = summary[budget][method]
            p = paper[budget][method]
            print(f"  {budget:>6}  {method:<12}  {m['rank_distance']:>12.3f}  {m['total_variation']:>12.3f}  {m['saturation']:>8.3f}  |  {p[0]:>12.1f}  {p[1]:>12.1f}  {p[2]:>8.2f}")

    # Write evaluation output
    output = {"summary": summary, "budgets": BUDGETS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Write scores.json (flat format matching convert_to_scores.py output)
    scores = {
        "experiments": {
            "table2_ablation_comparison": {
                "results": {}
            }
        }
    }
    for budget in BUDGETS:
        for method in ["random", "random_irt", "fluid"]:
            method_name = f"{method}_{budget}"
            m = summary[budget][method]
            scores["experiments"]["table2_ablation_comparison"]["results"][method_name] = {
                "rank_distance": m["rank_distance"],
                "total_variation": m["total_variation"],
                "saturation": m["saturation"],
            }

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.scores_output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved to {args.scores_output}")


if __name__ == "__main__":
    main()