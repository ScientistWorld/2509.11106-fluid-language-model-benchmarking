"""
Compute FLUID BENCHMARKING metrics from pre-computed upstream results.

This script loads the pre-computed ability estimates from the upstream
experiments.jsonl and computes the three evaluation dimensions:
- Validity: cross-benchmark rank distance
- Variance: normalized total variation of training curves
- Saturation: Spearman rank correlation of training curves

Uses the paper's Table 2 structure for comparison.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# Paths
UPSTREAM_RESULTS = Path("/home/user/tmp_fluid_upstream/results/experiments.jsonl")
OUTPUT_PATH = Path("/home/user/scoring/scores.json")

# Benchmark pairs for cross-benchmark validity
VALIDITY_PAIRS = [
    ("arc_challenge", "mmlu"),        # knowledge/reasoning
    ("hellaswag", "winogrande"),       # commonsense reasoning
]

# Budget sizes to evaluate (matching Table 2)
BUDGETS = [10, 50, 100, 500]

# Total checkpoints per LM (from upstream data)
LM_CHECKPOINT_COUNTS = {
    "amber-7b": 73,
    "k2-65b": 46,
    "olmo1-7b": 83,
    "olmo2-7b": 94,
    "pythia-3b": 78,
    "pythia-7b": 78,
}


def load_upstream_results() -> list[dict]:
    """Load pre-computed results from upstream experiments.jsonl."""
    rows = []
    with open(UPSTREAM_RESULTS) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_curves(rows: list[dict]) -> dict:
    """
    Build score curves from upstream results.

    Returns:
        curves[method][budget][(lm, benchmark)] = {checkpoint: score}
    """
    # Group rows by (lm, benchmark)
    by_lm_bm = defaultdict(dict)
    for row in rows:
        by_lm_bm[(row["lm"], row["benchmark"])][row["checkpoint"]] = row

    curves = {
        method: {budget: {} for budget in BUDGETS}
        for method in ["random", "random_irt", "fluid"]
    }

    for (lm, benchmark), checkpoints_data in by_lm_bm.items():
        sorted_cps = sorted(checkpoints_data.keys())

        for checkpoint in sorted_cps:
            row = checkpoints_data[checkpoint]

            # RANDOM (accuracy) and RANDOM_IRT (ability)
            for budget in BUDGETS:
                acc_key = f"random_accuracy_{budget}"
                abil_key = f"random_ability_{budget}"

                acc_val = row.get(acc_key)
                if acc_val is not None:
                    key = (lm, benchmark)
                    curves["random"][budget].setdefault(key, {})[checkpoint] = acc_val

                abil_val = row.get(abil_key)
                if abil_val is not None:
                    key = (lm, benchmark)
                    curves["random_irt"][budget].setdefault(key, {})[checkpoint] = abil_val

            # FLUID: abilities_fb is a list
            abilities = row.get("abilities_fb", [])
            for budget in BUDGETS:
                if abilities and len(abilities) >= budget:
                    key = (lm, benchmark)
                    curves["fluid"][budget].setdefault(key, {})[checkpoint] = abilities[budget - 1]

    return curves


def compute_total_variation(scores: list[float]) -> float:
    """
    Compute normalized total variation (Equation 6 in paper).

    TV = (n/(n-1)) * sum(|x^{t+1} - x^t|) / |x^n - x^1|

    Lower is better. Caps at 100 to handle the near-zero-range edge case
    (which would otherwise produce infinity).
    """
    x = np.array(scores, dtype=float)
    n = len(x)

    if n < 2:
        return 0.0

    total_variation = np.sum(np.abs(np.diff(x)))
    range_val = abs(x[-1] - x[0])

    if range_val < 1e-10:
        if total_variation < 1e-10:
            return 0.0
        # Near-zero range: cap at 100
        return 100.0

    tv = (n / (n - 1)) * total_variation / range_val
    return min(float(tv), 100.0)


def compute_saturation(scores: list[float]) -> float:
    """
    Compute saturation as monotonicity (Spearman rank correlation).
    Higher is better (1 = perfectly monotonic).
    """
    x = np.array(scores, dtype=float)
    n = len(x)

    if n < 2:
        return 1.0

    if len(set(x)) <= 1:
        return 0.0

    checkpoints = np.arange(n)
    corr, _ = spearmanr(checkpoints, x)
    return float(abs(corr))


def compute_rank_distance_from_arrays(src_vals, tgt_vals):
    """
    Compute mean absolute rank distance between two arrays.
    Rank 1 = highest value.
    """
    n = len(src_vals)
    if n < 2:
        return 0.0

    src_vals = np.array(src_vals)
    tgt_vals = np.array(tgt_vals)

    # Rank 1 = highest
    src_ranks = n - np.argsort(np.argsort(src_vals))
    tgt_ranks = n - np.argsort(np.argsort(tgt_vals))

    return float(np.mean(np.abs(src_ranks - tgt_ranks)))


def compute_validity(curves: dict, budget: int, method: str) -> float:
    """
    Compute cross-benchmark validity (rank distance).

    For each validity pair (A, B), compute how well estimated performance on A
    predicts actual accuracy on B. Uses rank distance.

    Specifically: rank all checkpoints (across all LMs) by their score on
    source benchmark A, compare to their rank by accuracy on target benchmark B.
    """
    distances = []

    for bm_a, bm_b in VALIDITY_PAIRS:
        # Both directions: A→B and B→A
        for src_bm, tgt_bm in [(bm_a, bm_b), (bm_b, bm_a)]:
            # Collect all checkpoint scores for source and target benchmarks
            # Source: use the method's score
            # Target: always use full accuracy (random_accuracy_{budget} with budget)

            # Group by LM to ensure we have matching checkpoints
            for lm in LM_CHECKPOINT_COUNTS:
                src_key = (lm, src_bm)
                tgt_key = (lm, tgt_bm)

                src_data = curves[method][budget].get(src_key, {})
                tgt_data = curves["random"][budget].get(tgt_key, {})  # accuracy on target

                # Get common checkpoints
                common_cps = sorted(set(src_data.keys()) & set(tgt_data.keys()))
                if len(common_cps) < 2:
                    continue

                src_vals = [src_data[cp] for cp in common_cps]
                tgt_vals = [tgt_data[cp] for cp in common_cps]

                rd = compute_rank_distance_from_arrays(src_vals, tgt_vals)
                distances.append(rd)

    return float(np.mean(distances)) if distances else 0.0


def compute_metrics(curves: dict) -> dict:
    """
    Compute all metrics for all methods and budgets.
    """
    summary = {}

    for budget in BUDGETS:
        summary[budget] = {}

        for method in ["random", "random_irt", "fluid"]:
            # --- Total variation and saturation ---
            tv_values = []
            sat_values = []

            for (lm, benchmark), checkpoint_scores in curves[method][budget].items():
                if len(checkpoint_scores) < 2:
                    continue

                sorted_cps = sorted(checkpoint_scores.keys())
                scores = [checkpoint_scores[cp] for cp in sorted_cps]

                tv_values.append(compute_total_variation(scores))
                sat_values.append(compute_saturation(scores))

            mean_tv = float(np.mean(tv_values)) if tv_values else 0.0
            mean_sat = float(np.mean(sat_values)) if sat_values else 0.0

            summary[budget][method] = {
                "total_variation": mean_tv,
                "saturation": mean_sat,
            }

        # --- Validity (rank distance) ---
        for method in ["random", "random_irt", "fluid"]:
            rd = compute_validity(curves, budget, method)
            summary[budget][method]["rank_distance"] = rd

    return summary


def main() -> None:
    print("Loading upstream results...")
    rows = load_upstream_results()
    print(f"  Loaded {len(rows)} rows")

    print("Building score curves...")
    curves = build_curves(rows)
    print("  Built curves for all methods and budgets")

    # Verify data
    for method in ["random", "random_irt", "fluid"]:
        for budget in BUDGETS:
            n_entries = sum(len(v) for v in curves[method][budget].values())
            print(f"  {method}/{budget}: {n_entries} checkpoint-benchmark entries")

    print("\nComputing metrics...")
    summary = compute_metrics(curves)

    print("\n" + "=" * 70)
    print("RESULTS vs PAPER TABLE 2")
    print("=" * 70)

    # Paper reference values for comparison
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
            match_indicator = "~" if abs(m["rank_distance"] - p[0]) < 5 else "!"
            print(f"  {budget:>6}  {method:<12}  {m['rank_distance']:>12.3f}  {m['total_variation']:>12.3f}  {m['saturation']:>8.3f}  |  {p[0]:>12.1f}  {p[1]:>12.1f}  {p[2]:>8.2f}")

    # Write scores.json
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
                "rank_distance": round(m["rank_distance"], 4),
                "total_variation": round(m["total_variation"], 4),
                "saturation": round(m["saturation"], 4),
            }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nWrote scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
