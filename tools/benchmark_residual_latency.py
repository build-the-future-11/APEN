#!/usr/bin/env python3
"""Measure prototype engineering latency on deterministic synthetic inputs.

No training, protected data, or forecast-accuracy evaluation is performed.
The exact baseline source is supplied locally; this script makes no network calls.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

CANDIDATE_BLOB = "5b7b2b5efc55c214834456e7f04c4d105b039f04"
BASELINE_BLOB = "13145e5636994a85e99a22dcbabada9c839f66fe"
CANDIDATE_COMMIT = "59bd5dfbe96980d3f4f365b604fec2b88b8caf6a"
BASELINE_COMMIT = "86f7d6c0cbf34c3f767f3271dcb4e953ac32d092"
CASES = ((32, 4, 8), (32, 32, 8), (256, 4, 8),
         (256, 32, 8), (1024, 4, 64), (1024, 32, 64))


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_source(path: Path, name: str, expected_blob: str):
    raw = path.read_bytes()
    actual = git_blob(raw)
    if actual != expected_blob:
        raise ValueError(f"Source identity mismatch for {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.EpisodicResidualMemory


def cpu_description() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unreported"


def run_case(baseline_cls, candidate_cls, capacity, top_k, key_dim, args):
    rng = np.random.default_rng(20261109 + capacity + key_dim + top_k)
    keys = rng.normal(size=(capacity, key_dim))
    residuals = rng.normal(size=capacity)
    queries = rng.normal(size=(32, key_dim))
    banks = {}
    for name, cls in (("baseline", baseline_cls), ("candidate", candidate_cls)):
        bank = cls(key_dim=key_dim, capacity=capacity, top_k=top_k, gain=0.5)
        for key, residual in zip(keys, residuals):
            bank.observe(key=key, residual=float(residual),
                         available_at=0, observed_at=0)
        banks[name] = bank

    # Numerical agreement is only checked for this ordinary finite workload.
    # The old baseline is known to fail the distinct extreme-value regressions.
    agreement = []
    for query in queries:
        before = banks["baseline"].predict(key=query, baseline=10.0, at_time=1)
        after = banks["candidate"].predict(key=query, baseline=10.0, at_time=1)
        if (before.eligible_entries, before.used_entries) != (capacity, top_k):
            raise AssertionError("Baseline retrieval count changed")
        if (after.eligible_entries, after.used_entries) != (capacity, top_k):
            raise AssertionError("Candidate retrieval count changed")
        agreement.append(abs(before.corrected - after.corrected))
        if not np.isclose(before.corrected, after.corrected, rtol=1e-14, atol=1e-14):
            raise AssertionError("Ordinary-workload prediction disagreement")

    for bank in banks.values():
        for i in range(args.warmup):
            bank.predict(key=queries[i % len(queries)], baseline=10.0, at_time=1)

    samples = {"baseline": [], "candidate": []}
    # Measure complete predict calls, not just the accumulation fragment.
    # Alternate order; retain GC enabled in its default state.
    for repeat in range(args.repetitions):
        names = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for name in names:
            bank = banks[name]
            start = time.perf_counter_ns()
            for i in range(args.iterations):
                bank.predict(key=queries[i % len(queries)], baseline=10.0, at_time=1)
            elapsed = time.perf_counter_ns() - start
            samples[name].append(elapsed / args.iterations / 1000.0)

    timing = {}
    for name, values in samples.items():
        timing[name] = {
            "median_microseconds_per_prediction": statistics.median(values),
            "min_batch_mean_microseconds": min(values),
            "max_batch_mean_microseconds": max(values),
            "batch_mean_microseconds": values,
        }
    return {
        "capacity": capacity, "top_k": top_k, "key_dim": key_dim,
        "max_absolute_prediction_difference": max(agreement),
        "timing": timing,
        "median_candidate_to_baseline_ratio":
            timing["candidate"]["median_microseconds_per_prediction"]
            / timing["baseline"]["median_microseconds_per_prediction"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path,
                        help="Explicit PR #10 memory.py; defaults to the current runtime, which must match the pin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1 or args.repetitions < 3 or args.warmup < 0:
        parser.error("Require positive iterations, >=3 repetitions, and nonnegative warmup")
    root = Path(__file__).resolve().parents[1]
    candidate_path = args.candidate_file or (root / "apen_event_residual" / "memory.py")
    baseline_cls = load_source(args.baseline_file, "apen_latency_baseline", BASELINE_BLOB)
    candidate_cls = load_source(candidate_path, "apen_latency_candidate", CANDIDATE_BLOB)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = [run_case(baseline_cls, candidate_cls, *case, args) for case in CASES]
    record = {
        "kind": "synthetic_cpu_engineering_latency",
        "not_scientific_efficacy_evidence": True,
        "started_at_utc": started,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_memory_git_blob": CANDIDATE_BLOB,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_memory_git_blob": BASELINE_BLOB,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "cpu": cpu_description(),
            "visible_cpu_count": os.cpu_count(), "gc_enabled": gc.isenabled(),
            "timer": "time.perf_counter_ns",
        },
        "protocol": {
            "cases": len(CASES), "repetitions": args.repetitions,
            "predictions_per_batch": args.iterations, "warmup_per_implementation": args.warmup,
            "comparison_queries_per_case": 32, "alternating_order": True,
            "population_and_validation_excluded_from_timing": True,
            "timed_operation": "complete V1 predict call on an unchanged full bank",
            "summary": "median of batch means; not a per-request tail-latency percentile",
            "scope": "ordinary finite synthetic keys/residuals; no training or dataset evaluation",
        },
        "cases": results,
        "limitations": [
            "Single shared container and one run; no CPU affinity or frequency isolation.",
            "No latency acceptance threshold or deployment hardware was specified.",
            "V2 clock wrapper overhead, ingestion, allocation peaks and concurrency are not benchmarked.",
            "The old baseline is numerically invalid for retained extreme-value regressions.",
            "No claim about forecasting benefit or the historical spatial-PDE APEN implementation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    for row in results:
        b = row["timing"]["baseline"]["median_microseconds_per_prediction"]
        c = row["timing"]["candidate"]["median_microseconds_per_prediction"]
        print(f'{row["capacity"]:4}/{row["top_k"]:2}/{row["key_dim"]:2}: '
              f'{b:.1f} -> {c:.1f} us/predict ({c / b:.2f}x)')
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
