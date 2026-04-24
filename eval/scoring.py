"""
Scoring and result saving for FLUID BENCHMARKING evaluation.

Reads evaluation results and produces scores.json.
"""

import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES_PATH = REPO_ROOT / 'scoring' / 'scores.json'


def score_and_save(scores_dict, output_path=None):
    """
    Save scores to JSON file.

    Args:
        scores_dict: dict with structure {experiment: {method: {metric: value}}}
        output_path: path to save scores.json (default: scoring/scores.json under repo root)
    """
    if output_path is None:
        output_path = DEFAULT_SCORES_PATH
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(scores_dict, f, indent=2)

    print(f"Scores saved to {output_path}")
    return output_path


def load_scores(path=None):
    """Load scores from JSON file."""
    if path is None:
        path = DEFAULT_SCORES_PATH
    else:
        path = Path(path)

    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def format_scores_for_display(scores_dict):
    """Format scores for human-readable display."""
    lines = []
    for experiment, methods in scores_dict.items():
        lines.append(f"\n=== {experiment} ===")
        for method, metrics in methods.items():
            lines.append(f"  {method}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {metric}: {value:.4f}")
                else:
                    lines.append(f"    {metric}: {value}")
    return "\n".join(lines)
