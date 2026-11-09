# Residual accumulation latency: 9 November 2026

This record closes the previously unmeasured CPU-cost question for the prospective
residual-memory repair. It is an engineering measurement on generated finite
numbers, not evidence of forecasting benefit or a historical APEN reproduction.

## Source and protocol

The benchmark reads both implementations by exact Git blob identity before loading
them. Candidate memory code is `5b7b2b5efc55c214834456e7f04c4d105b039f04`, retained from
commit `59bd5dfbe96980d3f4f365b604fec2b88b8caf6a`; the old direct-dot baseline is
`13145e5636994a85e99a22dcbabada9c839f66fe`, from `86f7d6c0cbf34c3f767f3271dcb4e953ac32d092`.
The archived `baseline_memory.py` is byte-identical to that old source and remains
numerically invalid for the documented extreme-value cases. It is not an endorsed
alternative implementation.

Each case populates a full bank once with deterministic NumPy normal inputs and
uses 32 generated queries. It validates ordinary-output agreement, warms each
implementation for 30 calls, and times seven batches of 200 complete V1 predict
calls. Baseline/candidate timing order alternates. Population and validation are
outside the timer. Reported values are medians of batch means, not request-level
tail latency. All 192 ordinary query pairs had identical corrected float outputs.

Runtime: Python 3.12.14, NumPy 2.3.5,
AMD EPYC 9V74 80-Core Processor; one shared Linux container, nine visible CPUs, default GC
enabled. No affinity, frequency control or isolation was applied.

## Observed full-prediction latency

| Bank capacity | Top-k | Dimensions | Old baseline (microseconds) | Exact mean (microseconds) | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 8 | 124.2 | 173.9 | 1.40 |
| 32 | 32 | 8 | 155.0 | 375.8 | 2.42 |
| 256 | 4 | 8 | 1011.0 | 1050.0 | 1.04 |
| 256 | 32 | 8 | 1029.9 | 1269.8 | 1.23 |
| 1024 | 4 | 64 | 8505.2 | 8244.0 | 0.97 |
| 1024 | 32 | 64 | 8134.3 | 8892.5 | 1.09 |

The default-size case adds about 50 microseconds per prediction in this run.
Selecting all 32 neighbours adds about 221 microseconds. The difference is
consistent with additional rational arithmetic per selected neighbour, but the
measurement does not isolate that fragment from the rest of prediction.

At larger capacity, context-distance work dominates and shared-host variability
is material. The 0.97 ratio is not evidence that exact arithmetic makes prediction
faster: the corresponding batch ranges overlap broadly. Every batch mean is
preserved in `latency.json`, including outliers.

No latency target or deployment hardware was specified, so these measurements do
not confer a performance acceptance. V2 wrapper overhead, ingestion, peak memory,
concurrent use and extreme-value latency are not measured. The repair preserves
numerical correctness in its retained tests; reviewers can now judge its observed
cost explicitly.

## Reproduce

From the repository root:

```sh
python3 tools/benchmark_residual_latency.py \
  --baseline-file research/verification/residual_latency_20261109/baseline_memory.py \
  --output /tmp/apen-residual-latency.json
python3 -m pytest -q -W error tests/test_memory.py tests/test_memory_v2.py
```

The script has no network calls. Its output path is explicit so a rerun can retain
this recorded measurement. The source pins intentionally reject a different
implementation; a later implementation needs its own benchmark review and source
identity.

Local correctness verification against the unchanged candidate runtime and tests:
**40 tests and 78 subtests passed**, warnings treated as errors, 0.19 seconds.
The full local receipt is in `local_validation.json`.

## Hosted verification and adoption

The PR workflow now checks out the actual PR head explicitly and treats warnings
as errors, matching the local command. It retains the same standard runner,
read-only permissions, pinned actions and engineering-test scope. No workflow
dispatch or private allowance is used. Hosted status is recorded in the PR after
the exact head completes.

No model code, original evidence, protected experiment, manuscript, default branch,
merge or production release changed. Maintainer review and integration remain
open.
