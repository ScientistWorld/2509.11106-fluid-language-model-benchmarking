"""Convert evaluation results to scoring/scores.json format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    summary = data["summary"]
    budgets = data.get("budgets", [10, 50, 100, 500])

    scores = {
        "experiments": {
            "table2_ablation_comparison": {
                "results": {}
            }
        }
    }

    for budget in budgets:
        bk = str(budget)
        if bk not in summary:
            continue
        for method in ["random", "random_irt", "fluid"]:
            method_name = f"{method}_{budget}"
            metrics = summary[bk][method]
            scores["experiments"]["table2_ablation_comparison"]["results"][method_name] = {
                "rank_distance": metrics["rank_distance"],
                "total_variation": metrics["total_variation"],
                "saturation": metrics["saturation"],
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
