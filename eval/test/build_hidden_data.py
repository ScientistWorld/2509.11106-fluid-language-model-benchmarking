"""Build the hidden test response tables from the public release CSVs.

Pure standard library (runs on a login node without numpy). Called by
scripts/download.sh when eval/test/data/hidden_responses.json is missing.

The three held-out language models (K2-65B, Pythia-2.8B, Pythia-6.9B) are kept
only here, never under data/. Responses are binarized exactly like the paper's
released loader (`value >= 0.5`), aligned to the row order of each benchmark's
released IRT table, and bit-packed per checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IRT_DIR = REPO_ROOT / "data" / "downloads" / "fluid-benchmarking" / "irt_models"
OUTPUT = Path(__file__).resolve().parent / "data" / "hidden_responses.json"
HF_REVISION = "e0a11a3be90972ffaa12c6e63957cfd5be3a3259"
BENCHMARKS = ["arc_challenge", "gsm8k", "hellaswag", "truthfulqa_mc2", "winogrande", "mmlu"]
HIDDEN_LMS = {
    "k2-65b": "4b8afb03a180d5c790ff6209dba7e2045b052aa4129f2bbddad6f03e45ddd731",
    "pythia-3b": "dccbfb2ddc9f281fb98434e954e96734d7b1ac3c5368ff4a8838813dabd17dfe",
    "pythia-7b": "7cc9befb1178009eb6f5a352dbe6b27ea6db98c751f1e729f0c63a1b1e3b1efb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def irt_item_ids(benchmark: str) -> list[str]:
    with (IRT_DIR / f"{benchmark}.csv").open() as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row[0] for row in reader]


def pack_bits(bits: list[int]) -> str:
    padded = bits + [0] * ((-len(bits)) % 8)
    raw = bytearray()
    for start in range(0, len(padded), 8):
        byte = 0
        for bit in padded[start:start + 8]:
            byte = (byte << 1) | bit
        raw.append(byte)
    return base64.b64encode(bytes(raw)).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory with the downloaded hidden-LM CSVs")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    item_ids = {benchmark: irt_item_ids(benchmark) for benchmark in BENCHMARKS}
    payload = {
        "source": {
            "dataset": "huggingface.co/datasets/allenai/fluid-benchmarking",
            "revision": HF_REVISION,
            "files_sha256": {},
        },
        "binarize": "response = 1 if released value >= 0.5 else 0 (paper's released loader)",
        "benchmarks": {
            benchmark: {
                "n_items": len(ids),
                "item_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            }
            for benchmark, ids in item_ids.items()
        },
        "lms": {},
    }
    for lm, expected_hash in HIDDEN_LMS.items():
        path = args.csv_dir / f"{lm}.csv"
        actual = sha256(path)
        if actual != expected_hash:
            raise SystemExit(f"{path}: sha256 {actual} != pinned {expected_hash}")
        payload["source"]["files_sha256"][lm] = actual
        with path.open() as handle:
            reader = csv.reader(handle)
            checkpoints = next(reader)[1:]
            table = {row[0]: row[1:] for row in reader}
        lm_entry = {"checkpoints": checkpoints, "responses": {}}
        for benchmark in BENCHMARKS:
            rows = [table[item_id] for item_id in item_ids[benchmark]]
            packed = []
            for column in range(len(checkpoints)):
                bits = []
                for row in rows:
                    value = float(row[column])
                    if value != value:
                        raise SystemExit(f"{lm}/{benchmark}: NaN response")
                    bits.append(1 if value >= 0.5 else 0)
                packed.append(pack_bits(bits))
            lm_entry["responses"][benchmark] = packed
        payload["lms"][lm] = lm_entry
        print(f"packed {lm}: {len(checkpoints)} checkpoints")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, sha256 {sha256(args.output)})")


if __name__ == "__main__":
    main()
