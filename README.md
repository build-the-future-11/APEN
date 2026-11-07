# APEN

APEN is a research line investigating episodic/spatial residual correction for forecasting systems.

## Current repository state

This repository contains the historical APEN closeout records and a small,
separate **prospective residual-memory engineering prototype**. It is not a
complete executable release of the historical APEN forecasting architecture.

The prototype is in `apen_event_residual/`, with unit tests in `tests/`, package
metadata in `pyproject.toml`, and engineering contracts in `research/`. It takes
an externally supplied baseline prediction and applies a bounded nearest-context
residual correction. It contains no trained forecaster, scientific dataset,
checkpoint, or benchmark result.

The full implementation, experiment scripts, paper source, datasets,
checkpoints, and reproduction machinery referenced in the historical project
are not restored by this prototype. Its tests cannot validate those historical
results.

## Scientific status

The maintained closeout record is `FINAL_STATUS_2026-09-30.md`.

Its present conclusion is:

- the strong positive APEN-v2 claim is **not supported** by the currently documented evidence boundary;
- reduced validation-selected Burgers comparisons recorded in the project history left APEN last and winning 0/3 fresh seeds;
- historical/pre-repair evidence must not be pooled with repaired-valid evidence;
- diagnostic or smoke-scale results must not be promoted into paper-scale evidence;
- any new submission-grade positive claim requires a new, complete, valid, frozen experiment matrix.

This repository therefore records a **negative / falsification-first closeout**, not a submission-ready positive paper.

## Reproducibility boundary

The prototype below can be installed and checked independently. Commands,
benchmark tables, figures, and manuscript claims from the historical forecasting
implementation still cannot be reproduced from this repository alone.

Do not claim that:

- APEN is submission-ready;
- the full benchmark suite is present here;
- the implementation has been independently reproduced from this repository;
- historical results have been revalidated on the current architecture;
- absent files or scripts are part of this revision.

If the executable research package is restored later, it should be added with explicit provenance and a revision-linked evidence manifest before stronger reproducibility claims are made.

## Run the engineering prototype

Python 3.10 or newer is required. From a checkout of this repository:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q -W error tests/test_memory.py tests/test_memory_v2.py
```

Use `EpisodicResidualMemoryV2` for a forward chronological stream:

```python
from apen_event_residual.causal_memory_v2 import EpisodicResidualMemoryV2

memory = EpisodicResidualMemoryV2(key_dim=2, capacity=32, top_k=4, gain=0.5)
memory.observe(key=[0.0, 1.0], residual=4.0, available_at=0, observed_at=0)
result = memory.predict(key=[0.0, 1.0], baseline=10.0, at_time=1)
assert result.corrected == 12.0
```

At each clock tick, make all predictions before ingesting observations. Both the
label availability time and its actual ingestion time must be strictly earlier
than the forecast. V2 rejects backward queries and backdated ingestion; replay
the original event stream into a fresh instance when a historical prediction is
needed. The original `EpisodicResidualMemory` class is retained for compatibility
but cannot reconstruct states lost through FIFO eviction.

Context distances use a scaled Euclidean norm to preserve very small nonzero
separations and very large representable distances. Genuine distances outside
the float64 range are rejected. This arithmetic repair changes neither the
intended distance metric nor the scientific status of APEN; see
[the source correction note](research/DISTANCE_SCALING_FIX_20261105.md).

Engineering tests cover timestamp admission, FIFO capacity, copied keys,
chronological ordering, deterministic distance ties, and numerical weighting.
They use artificial fixtures and provide no evidence of forecasting benefit.

For the exact source/evidence boundary as of 7 November 2026 and the separate
scientific publication gate, see [repository evidence boundary](research/REPOSITORY_BOUNDARY_20261107.md).

## Closeout rule

Preserve the negative result honestly. A successor study may test a materially revised hypothesis, but it should be treated as a new study with its own protocol, evidence, and conclusion rather than as a rewrite of this closeout.
