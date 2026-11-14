"""FLUID BENCHMARKING as a per-session benchmarking procedure (protocol server).

The evaluator starts this process (through ``scripts/method.sh``) and runs one
session per (LM checkpoint, benchmark, item budget). In a session the method may
query up to ``budget`` items of the benchmark, one at a time, and must then
return a single score. Line protocol on stdin/stdout (logs go to stderr):

    method    -> HELLO fluid-gym-1 <ability|accuracy>
    evaluator -> BEGIN <benchmark> <budget>
    method    -> Q <item index>        (row index in the benchmark's IRT table)
    evaluator -> R <0|1>
    ...
    method    -> S <score>
    evaluator -> QUIT                  (after the last session)

Policies:
  fluid       paper's method: 2PL MAP ability estimation (Eq. 3) with items
              selected one by one by maximum Fisher information at the current
              estimate (Eqs. 4-5); starts at the mean reference-LM ability.
  random_irt  ablation: fixed random item subset per (benchmark, budget), MAP ability.
  random      ablation: fixed random item subset, accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

from method.fluid import vectorized_fisher_info
from method.irt import estimate_ability

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "downloads" / "fluid-benchmarking"
BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
PROTOCOL = "fluid-gym-1"


def load_item_parameters() -> dict:
    params = {}
    for benchmark in BENCHMARKS:
        with (DATA_DIR / "irt_models" / f"{benchmark}.csv").open() as handle:
            reader = csv.reader(handle)
            header = next(reader)
            a_col, b_col = header.index("a"), header.index("b")
            rows = list(reader)
        params[benchmark] = (
            np.array([float(row[a_col]) for row in rows]),
            np.array([float(row[b_col]) for row in rows]),
        )
    return params


def start_abilities() -> dict:
    """Mean reference-LM ability per benchmark (the released code's start ability)."""
    with (DATA_DIR / "open_llm_leaderboard_results.json").open() as handle:
        oll = json.load(handle)
    return {benchmark: float(np.mean(list(oll[benchmark]["ability"].values()))) for benchmark in BENCHMARKS}


class FluidPolicy:
    scale = "ability"

    def __init__(self, params: dict, starts: dict):
        self.params = params
        self.starts = starts

    def run(self, benchmark: str, budget: int, ask) -> float:
        a, b = self.params[benchmark]
        n_items = len(a)
        theta = self.starts[benchmark]
        available = np.ones(n_items, dtype=bool)
        selected: list[int] = []
        responses: list[float] = []
        for _ in range(min(budget, n_items)):
            info = vectorized_fisher_info(theta, a, b)
            info = np.where(available, info, -np.inf)
            index = int(np.argmax(info))
            available[index] = False
            selected.append(index)
            responses.append(float(ask(index)))
            theta = estimate_ability(np.array(responses), a[selected], b[selected], theta_init=theta)
        return float(theta)


class RandomSubsetPolicy:
    def __init__(self, params: dict, starts: dict, aggregate: str):
        self.params = params
        self.starts = starts
        self.aggregate = aggregate
        self.scale = "ability" if aggregate == "irt" else "accuracy"

    def subset(self, benchmark: str, budget: int) -> np.ndarray:
        n_items = len(self.params[benchmark][0])
        rng = np.random.default_rng([zlib.crc32(benchmark.encode()), budget])
        return np.sort(rng.choice(n_items, size=min(budget, n_items), replace=False))

    def run(self, benchmark: str, budget: int, ask) -> float:
        a, b = self.params[benchmark]
        subset = self.subset(benchmark, budget)
        responses = np.array([float(ask(int(index))) for index in subset])
        if self.aggregate == "accuracy":
            return float(responses.mean())
        return float(estimate_ability(responses, a[subset], b[subset], theta_init=self.starts[benchmark]))


def serve(policy) -> None:
    out = sys.stdout
    inp = sys.stdin

    def send(line: str) -> None:
        out.write(line + "\n")
        out.flush()

    def ask(index: int) -> int:
        send(f"Q {index}")
        reply = inp.readline().split()
        if len(reply) != 2 or reply[0] != "R":
            raise RuntimeError(f"unexpected evaluator reply {reply!r}")
        return int(reply[1])

    send(f"HELLO {PROTOCOL} {policy.scale}")
    for line in inp:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] != "BEGIN" or len(parts) != 3:
            raise RuntimeError(f"unexpected evaluator message {line!r}")
        score = policy.run(parts[1], int(parts[2]), ask)
        send(f"S {score!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmarking-method protocol server")
    parser.add_argument("--policy", choices=["fluid", "random_irt", "random"], default="fluid")
    args = parser.parse_args()
    params = load_item_parameters()
    starts = start_abilities()
    if args.policy == "fluid":
        policy = FluidPolicy(params, starts)
    elif args.policy == "random_irt":
        policy = RandomSubsetPolicy(params, starts, "irt")
    else:
        policy = RandomSubsetPolicy(params, starts, "accuracy")
    serve(policy)


if __name__ == "__main__":
    main()
