"""Hidden test evaluator for the FLUID benchmarking gym.

This evaluator owns everything on the comparison side and runs the submitted
method itself:

* The submitted method is the executable ``scripts/method.sh``. It is started as
  a subprocess and speaks a line protocol on stdin/stdout (see
  ``briefing/evaluation.md``). For every (held-out LM checkpoint, benchmark,
  item budget) session it may query up to ``budget`` distinct items; the
  evaluator answers each query with the checkpoint's binary response and the
  method returns one scalar score. The evaluator counts and enforces the budget.
* Held-out responses for K2-65B, Pythia-2.8B and Pythia-6.9B live only in
  ``eval/test/data/hidden_responses.json`` and are unreadable while the method
  runs. Ground truth (full-benchmark accuracy curves), the Open LLM Leaderboard
  reference population, the RANDOM / RANDOM IRT / full-benchmark baselines, the
  checkpoint roster and the item budgets are all fixed here.
* Every scored number is recomputed from the raw per-session scores.
* The method must be a stateless, deterministic per-session procedure: a random
  subset of sessions is replayed in a fresh process (before the main pass, in a
  different order) and must produce identical queries and scores.
* Any protocol violation, budget overrun, impossible value, crash, timeout,
  replay mismatch or modification of the public data files fails the run, and
  every scored metric of every experiment is then written at its worst value.

Self-contained: imports nothing from eval/train or method/.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HIDDEN_DATA = HERE / "data" / "hidden_responses.json"
OLL_REFERENCE = HERE / "data" / "oll_reference.json"
PUBLIC_DATA = REPO_ROOT / "data" / "downloads" / "fluid-benchmarking"
DEFAULT_OUTPUT = REPO_ROOT / "scoring" / "scores_test.json"
DETAILS_DIR = HERE / "output"
METHOD_CMD = ["bash", str(REPO_ROOT / "scripts" / "method.sh")]

SLICE = "test"
SESSION_ORDER_SEED = 917_203
REPLAY_FRACTION = 0.12
BASELINE_SEEDS = [0, 1, 2, 3, 4]
MESSAGE_TIMEOUT_S = 900.0
PROTOCOL = "fluid-gym-1"

BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
VALIDITY_PAIRS = [("arc_challenge", "mmlu"), ("hellaswag", "winogrande")]
BUDGETS = [10, 50, 100, 500]
WORST_LM_BUDGET = 100
APPENDIX_H_BUDGET = 500

TABLE2 = "table2_ablation_comparison"
APPENDIX_H = "appendix_h_full_accuracy_comparison"
WORST_LM = "appendix_f_table4_worst_lm_100"

# Worst attainable value of each metric (rank distance: ranks lie in 1..103
# against the 102 reference LMs; total variation is capped at 100).
WORST = {"rank_distance": 102.0, "total_variation": 100.0, "saturation": 0.0}

# Public files the submitted method reads. They are part of the fixed problem;
# a run that changes them fails.
PUBLIC_SHA256 = {
    "open_llm_leaderboard_results.json": "b15f1484cdba75c013e9cfba54ff884420b332923303d8ff19c604d201adf77c",
    "irt_models/arc_challenge.csv": "3b59923617dcaf712197c911be88c4e217a93872b57fd2ad750cfca195a20c1e",
    "irt_models/gsm8k.csv": "fa36c499035b7d98a31a8024217c6421d525a41d2e71ab05a54d3d621e877695",
    "irt_models/hellaswag.csv": "3028910b401f973a320062923f07c71a8f5f4d480c74eed5f82db9cdc4c54bfa",
    "irt_models/mmlu.csv": "a86a139b9138102aaa804a5b81d1d5cbaed8c8f05644073795dacc2929c9cc06",
    "irt_models/truthfulqa_mc2.csv": "385897f4ad000f1666846c342e102cac25480a2e7878bd041141b142ec721631",
    "irt_models/winogrande.csv": "8a6491475b2cfe9fd7aaf04eb2848af0bc7e11feda61b175aa92342f6897347d",
}
OLL_SHA256 = "b15f1484cdba75c013e9cfba54ff884420b332923303d8ff19c604d201adf77c"
HIDDEN_SHA256 = "66dd76b778a05f1d0e6872c95dda33d397bdf340123525629fcc4c5755617ded"


def set_mode(path: Path, mode: int) -> None:
    """Best-effort permission change (the hidden table is unreadable while the method runs)."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


class MethodFailure(Exception):
    """The submitted method violated the protocol or a constraint."""


class InfrastructureError(Exception):
    """The evaluator itself cannot run (missing or corrupt evaluator data)."""


# --------------------------------------------------------------------------
# Data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def public_file_hashes() -> dict:
    hashes = {}
    for rel in PUBLIC_SHA256:
        path = PUBLIC_DATA / rel
        hashes[rel] = sha256_file(path) if path.is_file() else None
    return hashes


def load_item_parameters() -> dict:
    params = {}
    for benchmark in BENCHMARKS:
        with (PUBLIC_DATA / "irt_models" / f"{benchmark}.csv").open() as handle:
            reader = csv.reader(handle)
            header = next(reader)
            a_col, b_col = header.index("a"), header.index("b")
            ids, a, b = [], [], []
            for row in reader:
                ids.append(row[0])
                a.append(float(row[a_col]))
                b.append(float(row[b_col]))
        params[benchmark] = {"ids": ids, "a": np.array(a), "b": np.array(b)}
    return params


def load_hidden(item_params: dict) -> dict:
    """Return {lm: {"checkpoints": [...], "responses": {bm: uint8[n_ckpt, n_items]}}}."""
    set_mode(HIDDEN_DATA, 0o640)
    if sha256_file(HIDDEN_DATA) != HIDDEN_SHA256:
        raise InfrastructureError("hidden response table does not match its pinned sha256")
    payload = json.loads(HIDDEN_DATA.read_text())
    for benchmark in BENCHMARKS:
        expected = payload["benchmarks"][benchmark]
        ids = item_params[benchmark]["ids"]
        if expected["n_items"] != len(ids) or expected["item_ids_sha256"] != hashlib.sha256("\n".join(ids).encode()).hexdigest():
            raise InfrastructureError(f"{benchmark}: item order differs from hidden table")
    data = {}
    for lm, entry in payload["lms"].items():
        responses = {}
        for benchmark in BENCHMARKS:
            n_items = payload["benchmarks"][benchmark]["n_items"]
            rows = [
                np.unpackbits(np.frombuffer(base64.b64decode(text), dtype=np.uint8))[:n_items]
                for text in entry["responses"][benchmark]
            ]
            responses[benchmark] = np.stack(rows).astype(np.uint8)
        data[lm] = {"checkpoints": list(entry["checkpoints"]), "responses": responses}
    return data


def load_reference_population() -> dict:
    if sha256_file(OLL_REFERENCE) != OLL_SHA256:
        raise InfrastructureError("evaluator copy of the reference population does not match its pinned sha256")
    raw = json.loads(OLL_REFERENCE.read_text())
    return {
        benchmark: {
            "accuracy": np.array(sorted(float(v) for v in raw[benchmark]["accuracy"].values())),
            "ability": np.array(sorted(float(v) for v in raw[benchmark]["ability"].values())),
        }
        for benchmark in BENCHMARKS
    }


# --------------------------------------------------------------------------
# Protocol runner


class MethodProcess:
    def __init__(self, log_path: Path, workdir: Path):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
        env["TMPDIR"] = str(workdir)
        self.log_handle = log_path.open("ab")
        self.proc = subprocess.Popen(
            METHOD_CMD,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
        )
        self.fd = self.proc.stdout.fileno()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.fd, selectors.EVENT_READ)
        self.buffer = b""
        self.scale = None

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write((line + "\n").encode("ascii"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MethodFailure(f"method process closed its input ({exc})") from exc

    def readline(self) -> str:
        while b"\n" not in self.buffer:
            if not self.selector.select(timeout=MESSAGE_TIMEOUT_S):
                raise MethodFailure(f"method sent nothing for {MESSAGE_TIMEOUT_S:.0f} s")
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                try:
                    code = self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    code = None
                raise MethodFailure(f"method process exited (code {code}) before answering")
            self.buffer += chunk
            if len(self.buffer) > 1 << 20:
                raise MethodFailure("method wrote an over-long line to stdout")
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            return line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise MethodFailure("method wrote non-ASCII bytes to stdout") from exc

    def handshake(self) -> str:
        parts = self.readline().split()
        if len(parts) != 3 or parts[0] != "HELLO" or parts[1] != PROTOCOL or parts[2] not in ("ability", "accuracy"):
            raise MethodFailure(f"bad handshake {' '.join(parts)!r}; expected 'HELLO {PROTOCOL} ability|accuracy'")
        self.scale = parts[2]
        return self.scale

    def run_session(self, benchmark: str, budget: int, responses: np.ndarray) -> tuple[float, list[int]]:
        n_items = responses.shape[0]
        self.send(f"BEGIN {benchmark} {budget}")
        queried: list[int] = []
        seen = set()
        while True:
            parts = self.readline().split()
            if len(parts) == 2 and parts[0] == "Q":
                try:
                    index = int(parts[1])
                except ValueError as exc:
                    raise MethodFailure(f"non-integer item index {parts[1]!r}") from exc
                if not 0 <= index < n_items:
                    raise MethodFailure(f"{benchmark}: item index {index} outside 0..{n_items - 1}")
                if index in seen:
                    raise MethodFailure(f"{benchmark}: item {index} queried twice in one session")
                if len(queried) >= budget:
                    raise MethodFailure(f"{benchmark}: query beyond the {budget}-item budget")
                seen.add(index)
                queried.append(index)
                self.send(f"R {int(responses[index])}")
            elif len(parts) == 2 and parts[0] == "S":
                try:
                    value = float(parts[1])
                except ValueError as exc:
                    raise MethodFailure(f"non-numeric score {parts[1]!r}") from exc
                check_score(value, self.scale)
                return value, queried
            else:
                raise MethodFailure(f"unexpected message {' '.join(parts)[:80]!r}")

    def close(self) -> None:
        try:
            self.send("QUIT")
        except MethodFailure:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.selector.close()
        self.log_handle.close()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.selector.close()
        self.log_handle.close()


def check_score(value: float, scale: str) -> None:
    if not math.isfinite(value):
        raise MethodFailure(f"non-finite score {value}")
    if scale == "accuracy" and not 0.0 <= value <= 1.0:
        raise MethodFailure(f"accuracy-scale score {value} outside [0, 1]")
    if scale == "ability" and abs(value) > 1e6:
        raise MethodFailure(f"ability-scale score {value} is not a plausible ability")


def run_sessions(sessions: list, order: list, hidden: dict, log_path: Path, workdir: Path, label: str) -> tuple[str, dict]:
    process = MethodProcess(log_path, workdir)
    outputs = {}
    try:
        scale = process.handshake()
        started = time.time()
        for count, index in enumerate(order, 1):
            lm, checkpoint_index, benchmark, budget = sessions[index]
            responses = hidden[lm]["responses"][benchmark][checkpoint_index]
            outputs[index] = process.run_session(benchmark, budget, responses)
            if count % 1000 == 0:
                print(f"  [{label}] {count}/{len(order)} sessions ({time.time() - started:.0f} s)", flush=True)
        process.close()
    except BaseException:
        process.kill()
        raise
    return scale, outputs


def run_method(hidden: dict, workdir: Path, log_path: Path) -> tuple[str, dict, dict]:
    sessions = []
    for lm in sorted(hidden):
        for checkpoint_index in range(len(hidden[lm]["checkpoints"])):
            for benchmark in BENCHMARKS:
                for budget in BUDGETS:
                    sessions.append((lm, checkpoint_index, benchmark, budget))
    rng = np.random.default_rng(SESSION_ORDER_SEED)
    main_order = [int(i) for i in rng.permutation(len(sessions))]
    n_replay = max(1, int(round(REPLAY_FRACTION * len(sessions))))
    replay_set = [int(i) for i in rng.choice(len(sessions), size=n_replay, replace=False)]
    replay_order = [int(i) for i in rng.permutation(replay_set)]

    print(f"Replay pass: {len(replay_order)} sessions in a fresh method process", flush=True)
    replay_scale, replay_outputs = run_sessions(sessions, replay_order, hidden, log_path, workdir, "replay")
    print(f"Main pass: {len(main_order)} sessions", flush=True)
    scale, outputs = run_sessions(sessions, main_order, hidden, log_path, workdir, "main")

    if replay_scale != scale:
        raise MethodFailure("method declared different score scales in two processes")
    mismatches = 0
    for index in replay_set:
        value_a, items_a = replay_outputs[index]
        value_b, items_b = outputs[index]
        if items_a != items_b or abs(value_a - value_b) > 1e-9 * max(1.0, abs(value_a)):
            mismatches += 1
    if mismatches:
        raise MethodFailure(
            f"{mismatches}/{len(replay_set)} replayed sessions differ from the main pass: the method must be a "
            "deterministic, stateless per-session procedure"
        )

    curves = {}
    queries = {}
    for index, (lm, checkpoint_index, benchmark, budget) in enumerate(sessions):
        value, items = outputs[index]
        curves.setdefault(budget, {}).setdefault(lm, {}).setdefault(
            benchmark, np.zeros(len(hidden[lm]["checkpoints"]))
        )[checkpoint_index] = value
        queries.setdefault(budget, []).append(len(items))
    stats = {
        "sessions": len(sessions),
        "replayed_sessions": len(replay_set),
        "mean_items_queried": {str(budget): float(np.mean(counts)) for budget, counts in sorted(queries.items())},
    }
    return scale, curves, stats


# --------------------------------------------------------------------------
# Metrics


def total_variation(curve: np.ndarray) -> float:
    n = len(curve)
    if n < 2:
        return WORST["total_variation"]
    steps = float(np.sum(np.abs(np.diff(curve))))
    span = abs(float(curve[-1]) - float(curve[0]))
    if span < 1e-10:
        # Equation 6 is undefined; a curve whose end equals its start (including
        # a constant curve) carries no training signal.
        return WORST["total_variation"]
    return min(n / (n - 1) * steps / span, WORST["total_variation"])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start:end + 1]] = 0.5 * (start + end)
        start = end + 1
    return ranks


def saturation(curve: np.ndarray) -> float:
    if len(curve) < 2 or np.all(curve == curve[0]):
        return WORST["saturation"]
    ranks = average_ranks(np.asarray(curve, dtype=float))
    steps = np.arange(len(curve), dtype=float)
    ranks = ranks - ranks.mean()
    steps = steps - steps.mean()
    denom = math.sqrt(float(np.sum(ranks ** 2) * np.sum(steps ** 2)))
    return abs(float(np.sum(ranks * steps)) / denom) if denom > 0 else WORST["saturation"]


def reference_rank(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """1 + number of reference LMs scoring strictly higher."""
    return 1 + len(sorted_reference) - np.searchsorted(sorted_reference, values, side="right")


def validity_distances(curves_lm: dict, full_accuracy_lm: dict, scale: str, population: dict) -> np.ndarray:
    distances = []
    for first, second in VALIDITY_PAIRS:
        for source, target in ((first, second), (second, first)):
            source_rank = reference_rank(curves_lm[source], population[source][scale])
            target_rank = reference_rank(full_accuracy_lm[target], population[target]["accuracy"])
            distances.append(np.abs(source_rank - target_rank).astype(float))
    return np.concatenate(distances)


def curve_metrics(curves: dict, full_accuracy: dict, scale: str, population: dict, lms: list) -> dict:
    distances = np.concatenate([validity_distances(curves[lm], full_accuracy[lm], scale, population) for lm in lms])
    tvs = [total_variation(curves[lm][benchmark]) for lm in lms for benchmark in BENCHMARKS]
    sats = [saturation(curves[lm][benchmark]) for lm in lms for benchmark in BENCHMARKS]
    return {
        "rank_distance": float(np.mean(distances)),
        "total_variation": float(np.mean(tvs)),
        "saturation": float(np.mean(sats)),
    }


def worst_lm_metrics(curves: dict, full_accuracy: dict, scale: str, population: dict, lms: list) -> tuple[dict, dict]:
    per_lm = {lm: curve_metrics(curves, full_accuracy, scale, population, [lm]) for lm in lms}
    worst = {
        "rank_distance": max(m["rank_distance"] for m in per_lm.values()),
        "total_variation": max(m["total_variation"] for m in per_lm.values()),
        "saturation": min(m["saturation"] for m in per_lm.values()),
    }
    return worst, per_lm


# --------------------------------------------------------------------------
# Evaluator-owned baselines


def map_ability(responses: np.ndarray, a: np.ndarray, b: np.ndarray, low=-4.0, high=4.0, iterations=60) -> np.ndarray:
    """MAP ability (N(0,1) prior, clipped to [low, high]) for each row of responses.

    The posterior score is strictly decreasing in theta, so bisection finds the
    unique root; rows whose root lies outside the range are clipped.
    """
    responses = responses.astype(float)

    def score(theta):
        p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
        return -theta + np.sum(a[None, :] * (responses - p), axis=1)

    rows = responses.shape[0]
    lo = np.full(rows, low)
    hi = np.full(rows, high)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        positive = score(mid) > 0
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)
    theta = 0.5 * (lo + hi)
    theta = np.where(score(np.full(rows, low)) <= 0, low, theta)
    theta = np.where(score(np.full(rows, high)) >= 0, high, theta)
    return theta


def baseline_curves(hidden: dict, item_params: dict) -> dict:
    curves = {"random": {}, "random_irt": {}}
    for seed in BASELINE_SEEDS:
        for benchmark_index, benchmark in enumerate(BENCHMARKS):
            n_items = len(item_params[benchmark]["ids"])
            a, b = item_params[benchmark]["a"], item_params[benchmark]["b"]
            for budget in BUDGETS:
                rng = np.random.default_rng([seed, benchmark_index, budget])
                subset = np.sort(rng.choice(n_items, size=budget, replace=False))
                for lm in hidden:
                    responses = hidden[lm]["responses"][benchmark][:, subset]
                    curves["random"].setdefault(seed, {}).setdefault(budget, {}).setdefault(lm, {})[benchmark] = responses.mean(axis=1)
                    curves["random_irt"].setdefault(seed, {}).setdefault(budget, {}).setdefault(lm, {})[benchmark] = map_ability(
                        responses, a[subset], b[subset]
                    )
    return curves


def compute_baselines(hidden: dict, item_params: dict, full_accuracy: dict, population: dict) -> dict:
    lms = sorted(hidden)
    curves = baseline_curves(hidden, item_params)
    table2, worst_lm, per_seed, per_lm = {}, {}, {}, {}
    for name, scale in (("random", "accuracy"), ("random_irt", "ability")):
        for budget in BUDGETS:
            seed_metrics = [curve_metrics(curves[name][seed][budget], full_accuracy, scale, population, lms) for seed in BASELINE_SEEDS]
            table2[f"{name}_{budget}"] = {metric: float(np.mean([m[metric] for m in seed_metrics])) for metric in WORST}
            per_seed[f"{name}_{budget}"] = seed_metrics
        seed_results = [worst_lm_metrics(curves[name][seed][WORST_LM_BUDGET], full_accuracy, scale, population, lms) for seed in BASELINE_SEEDS]
        worst_lm[f"{name}_{WORST_LM_BUDGET}"] = {metric: float(np.mean([w[metric] for w, _ in seed_results])) for metric in WORST}
        for lm in lms:
            per_lm.setdefault(lm, {})[name] = {
                metric: float(np.mean([p[lm][metric] for _, p in seed_results])) for metric in WORST
            }

    full_ability_curves = {
        lm: {
            benchmark: map_ability(hidden[lm]["responses"][benchmark], item_params[benchmark]["a"], item_params[benchmark]["b"])
            for benchmark in BENCHMARKS
        }
        for lm in lms
    }
    appendix = {
        "full_accuracy": curve_metrics(full_accuracy, full_accuracy, "accuracy", population, lms),
        "full_ability": curve_metrics(full_ability_curves, full_accuracy, "ability", population, lms),
    }
    return {"table2": table2, "worst_lm": worst_lm, "appendix_h": appendix, "per_seed": per_seed, "per_lm": per_lm}


# --------------------------------------------------------------------------
# Score files


def worst_scores(reason: str) -> dict:
    worst = dict(WORST)
    return {
        "experiments": {
            TABLE2: {"results": {f"submitted_{budget}": dict(worst) for budget in BUDGETS}},
            APPENDIX_H: {"results": {f"submitted_{APPENDIX_H_BUDGET}": dict(worst)}},
            WORST_LM: {"results": {f"submitted_{WORST_LM_BUDGET}": dict(worst)}},
        },
        "status": "failed",
        "failure_reason": reason,
        "slice": SLICE,
    }


def build_scores(scale: str, curves: dict, full_accuracy: dict, population: dict, baselines: dict, lms: list, stats: dict) -> tuple[dict, dict]:
    table2 = {f"submitted_{budget}": curve_metrics(curves[budget], full_accuracy, scale, population, lms) for budget in BUDGETS}
    worst, per_lm = worst_lm_metrics(curves[WORST_LM_BUDGET], full_accuracy, scale, population, lms)
    scores = {
        "experiments": {
            TABLE2: {
                "results": table2,
                "baselines": baselines["table2"],
            },
            APPENDIX_H: {
                "results": {f"submitted_{APPENDIX_H_BUDGET}": dict(table2[f"submitted_{APPENDIX_H_BUDGET}"])},
                "baselines": baselines["appendix_h"],
            },
            WORST_LM: {
                "results": {f"submitted_{WORST_LM_BUDGET}": worst},
                "baselines": baselines["worst_lm"],
                "per_lm": {lm: {"submitted": per_lm[lm], **baselines["per_lm"][lm]} for lm in lms},
            },
        },
        "status": "ok",
        "slice": SLICE,
        "lms": lms,
        "score_scale": scale,
        "evaluation": stats,
    }
    details = {
        "scale": scale,
        "lms": lms,
        "checkpoints": None,
        "curves": {str(budget): {lm: {bm: curves[budget][lm][bm].tolist() for bm in BENCHMARKS} for lm in lms} for budget in BUDGETS},
        "baseline_per_seed": baselines["per_seed"],
    }
    return scores, details


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DETAILS_DIR / "method_stderr.log"
    log_path.write_bytes(b"")

    try:
        before = public_file_hashes()
        mismatched = [rel for rel, digest in before.items() if digest != PUBLIC_SHA256[rel]]
        if any(before[rel] is None for rel in before):
            raise InfrastructureError("public data missing; run scripts/download.sh")
        item_params = load_item_parameters()
        population = load_reference_population()
        hidden = load_hidden(item_params)
    except InfrastructureError as exc:
        print(f"EVALUATOR ERROR: {exc}", file=sys.stderr)
        return 2

    lms = sorted(hidden)
    full_accuracy = {lm: {bm: hidden[lm]["responses"][bm].mean(axis=1) for bm in BENCHMARKS} for lm in lms}
    baselines = compute_baselines(hidden, item_params, full_accuracy, population)

    workdir = Path(tempfile.mkdtemp(prefix="fluid_gym_method_"))
    started = time.time()
    try:
        if mismatched:
            raise MethodFailure(f"public data files were modified before evaluation: {mismatched}")
        set_mode(HIDDEN_DATA, 0o000)
        try:
            scale, curves, stats = run_method(hidden, workdir, log_path)
        finally:
            set_mode(HIDDEN_DATA, 0o640)
        after = public_file_hashes()
        changed = [rel for rel in PUBLIC_SHA256 if after[rel] != PUBLIC_SHA256[rel]]
        if changed:
            raise MethodFailure(f"method modified public data files during evaluation: {changed}")
        stats["method_seconds"] = round(time.time() - started, 1)
        scores, details = build_scores(scale, curves, full_accuracy, population, baselines, lms, stats)
        details["checkpoints"] = {lm: hidden[lm]["checkpoints"] for lm in lms}
        write_json(DETAILS_DIR / "details.json", details)
    except MethodFailure as exc:
        print(f"RUN FAILED: {exc}", flush=True)
        scores = worst_scores(str(exc))
    except Exception as exc:  # any crash while running the submitted method
        traceback.print_exc()
        print(f"RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
        scores = worst_scores(f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if scores["status"] == "failed":
        scores["experiments"][TABLE2]["baselines"] = baselines["table2"]
        scores["experiments"][APPENDIX_H]["baselines"] = baselines["appendix_h"]
        scores["experiments"][WORST_LM]["baselines"] = baselines["worst_lm"]
    write_json(args.output, scores)
    print(f"Wrote {args.output} (status: {scores['status']})")
    for name, metrics in scores["experiments"][TABLE2]["results"].items():
        print(f"  {name:<14} rank_distance={metrics['rank_distance']:.4f} total_variation={metrics['total_variation']:.4f} saturation={metrics['saturation']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
