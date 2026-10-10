# APEN residual-memory event ordering V2

Status: prospective engineering correction, no scientific outcome.

## Reproduced defect

The PR4 prototype at `de1ba1217d3d78e2790840fcfe5c8b1697056457` filters
records by label availability and actual ingestion time. That alone is
insufficient for historical queries when capacity is bounded: a future record
can evict an earlier eligible record before the query filters the remaining bank.

The local artificial witness uses capacity 1, gain 1, key `[0]`, baseline 10:

1. Ingest residual 2 at tick 1; query tick 2 returns 12.
2. Ingest residual 100 at tick 10; this evicts the original entry.
3. Query tick 2 again: the legacy class returns 10.

The future residual is not retrieved, yet future ingestion changed the past
prediction through FIFO state. The same defect exists at equal ticks when a
same-tick ingestion evicts a prior record before a strictly-earlier-data query.

## Explicit V2 contract

Import `apen_event_residual.causal_memory_v2.EpisodicResidualMemoryV2`.
The old class and its nine original tests remain unchanged. V2 inherits the
original geometry, distance, weighting, gain, capacity, availability checks and
receipts, adding a bounded forward-only event-order contract:

- Prediction times cannot decrease.
- At each tick, perform all predictions before ingesting observations.
- Multiple predictions at a tick are allowed until ingestion begins at that tick.
- After ingestion at tick t, the next prediction must be strictly later than t.
- Actual ingestion times cannot decrease or predate a completed prediction.
- Labels may have become available earlier. Record actual ingestion time in
  `observed_at`; do not backdate a late-arriving record to its label timestamp.
- Rejected operations neither advance clocks nor change the bank.

Only two integer clock values are added; capacity remains fixed. There is no
historical archive. To reconstruct an earlier prediction, replay the original
event sequence in a fresh instance. A coarse clock that cannot represent the
needed event order should be refined; do not silently move event timestamps.
This API assumes one writer and does not claim thread safety or event-clock truth
verification. It verifies consistency of timestamps supplied by the caller.

## Bounded validation

Use tiny synthetic scalar keys and residuals. Run one local standard-library
unittest collection of the 14 new cases, followed by the existing safe hosted
prototype CI with the nine original cases plus the new cases. A corrective cycle
is permitted only for a concrete implementation or test failure. No training,
benchmark campaign, protected data, paid compute or scientific workflow dispatch.

```bash
python -m unittest discover -s tests -p test_memory_v2.py -v
python -m pytest -q tests/test_memory.py tests/test_memory_v2.py
```

The new suite keeps the original failure as an explicit witness and checks
future/same-tick eviction rejection, both clock directions, invalid-operation
atomicity, tick zero, grouped operations, capacity, exact agreement with the
legacy class on valid forward streams, and reconstruction by fresh replay.

The original APEN scientific line remains negative and its release/provenance
status remains unresolved. This control does not establish a new APEN architecture,
improved benchmark accuracy, repaired historical experiments or submission readiness.
The code requires independent review and a separately frozen scientific protocol
before any outcome-bearing use.
