"""Convert experiment outputs into scoring/scores.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE_PATH = Path("/home/user/scoring/reference.json")
EXPERIMENT_NAME = "table2_ablation_comparison"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def build_scores(payload: dict, reference: dict) -> dict:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Results payload is missing the computed 'summary' section")

    ref_experiment = reference["experiments"][EXPERIMENT_NAME]
    ref_methods = ref_experiment["results"]
    scores = {"experiments": {EXPERIMENT_NAME: {"results": {}}}}

    for budget, method_metrics in summary.items():
        for method, metrics in method_metrics.items():
            method_name = f"{method}_{budget}"
            if method_name not in ref_methods:
                continue
            scores["experiments"][EXPERIMENT_NAME]["results"][method_name] = {
                metric_name: float(metrics[metric_name])
                for metric_name in ref_experiment["metrics"]
                if metric_name in metrics
            }

    if not scores["experiments"][EXPERIMENT_NAME]["results"]:
        raise ValueError("No experiment results matched the reference methods")

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/home/user/checkpoints/experiment_results.json"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/user/scoring/scores.json"),
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
