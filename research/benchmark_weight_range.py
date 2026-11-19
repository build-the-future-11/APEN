"""Compare local prototype prediction latency against an explicit source file.

Artificial inputs only. This measures CPU overhead, not forecasting efficacy.
Run from the repository root with PYTHONPATH=. and --baseline-source pointing
to memory.py recovered from the reference commit recorded in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from apen_event_residual.memory import EpisodicResidualMemory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-source', required=True, type=Path)
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('weight_range_baseline', args.baseline_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old_cls = module.EpisodicResidualMemory
    rng = np.random.default_rng(92742)
    records = []
    for capacity, top_k, iterations in ((32, 4, 120), (32, 32, 40),
                                       (128, 128, 8), (256, 32, 20)):
        keys = rng.normal(size=(capacity, 8))
        residuals = rng.normal(size=capacity)
        query = rng.normal(size=8)
        instances = {}
        for label, cls in (('parent', old_cls), ('candidate', EpisodicResidualMemory)):
            memory = cls(key_dim=8, capacity=capacity, top_k=top_k, gain=.5)
            for key, residual in zip(keys, residuals):
                memory.observe(key=key, residual=float(residual), available_at=0, observed_at=0)
            for _ in range(10):
                memory.predict(key=query, baseline=1., at_time=1)
            instances[label] = memory
        samples = {'parent': [], 'candidate': []}
        for repeat in range(7):
            order = ('parent', 'candidate') if repeat % 2 == 0 else ('candidate', 'parent')
            for label in order:
                memory = instances[label]
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    result = memory.predict(key=query, baseline=1., at_time=1)
                samples[label].append((time.perf_counter_ns() - started) / iterations / 1000)
        medians = {label: statistics.median(values) for label, values in samples.items()}
        outputs = {label: memory.predict(key=query, baseline=1., at_time=1).corrected
                   for label, memory in instances.items()}
        records.append({'capacity': capacity, 'top_k': top_k, 'key_dim': 8,
                        'iterations_per_batch': iterations, 'batches': 7,
                        'microseconds_per_prediction': samples, 'median_microseconds': medians,
                        'candidate_over_parent': medians['candidate'] / medians['parent'],
                        'absolute_output_difference': abs(outputs['candidate'] - outputs['parent'])})
    candidate_path = Path(sys.modules[EpisodicResidualMemory.__module__].__file__)
    output = {'scope': 'local artificial-input engineering timing; not a scientific efficacy result',
              'reference_commit': args.baseline_commit,
              'source_sha256': {'parent': hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                                'candidate': hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
              'environment': {'python': sys.version, 'numpy': np.__version__,
                              'platform': platform.platform(), 'processor': platform.processor()},
              'warmup_predictions_per_case_and_implementation': 10,
              'order': 'alternating parent/candidate across seven batches',
              'seed': 92742, 'records': records,
              'limits': ['Single machine and one local run; medians are descriptive.',
                         'Repeated query per bank; not training or live-stream throughput.',
                         'No claimed upper bound for arbitrary top_k or adversarial denominators.',
                         'Gain multiplication, distance and baseline addition remain float arithmetic.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for row in records:
        print(f"capacity={row['capacity']} top_k={row['top_k']} "
              f"parent={row['median_microseconds']['parent']:.1f} us "
              f"candidate={row['median_microseconds']['candidate']:.1f} us "
              f"ratio={row['candidate_over_parent']:.2f}")


if __name__ == '__main__':
    main()
