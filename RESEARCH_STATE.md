# APEN — RESEARCH_STATE

## Research question
Does episodic residual retrieval improve a spatial forecaster beyond strong matched alternatives under a valid causal protocol?

## Current scientific status
The current APEN line should be treated as a **NEGATIVE** result with respect to the strong superiority claim.

The retained manuscript history contains small improvements over a frozen base, but the strongest equal-update control favored continued base training. The current repository status further records that the repaired/current APEN comparison is not submission-ready and that reduced validation-selected comparisons do not support superiority.

## Repository/release state
The public default branch currently contains only `README.md` and `FINAL_STATUS_2026-09-30.md`.

The README references paths and commands including `APEN.md`, `STATUS.md`, `RESEARCH_TRUTH.md`, `REPRODUCE.md`, `scripts/preflight.py`, and `paper/apen_v2/README.md`, but those paths are absent from the current default branch.

Therefore the public repository is **not a reproducible release of the implementation described by its README**.

## Evidence boundary
- Historical/pre-repair evidence must remain separate from repaired-valid evidence.
- A polished manuscript is not a substitute for a reproducible current source tree.
- No frozen endpoint should be rerun merely to seek a positive result.
- Any new positive APEN claim requires a separately frozen successor experiment.

## Completion criteria for the current line
1. Recover and bind the canonical current source tree.
2. Restore only files that are provenance-valid for the current architecture.
3. Create explicit `legacy_contaminated/` and `repaired_valid/` evidence partitions (or equivalent manifest metadata).
4. Bind every paper number to raw evidence and source/config identity.
5. Make the negative/current claim boundary explicit in the manuscript and README.
6. Run clean reproduction only after canonical source identity is established.
7. Freeze a release hash and produce a completion manifest.

## Terminal disposition
**NEGATIVE — scientific claim resolved; reproducible release incomplete.**

## Prospective event-order V2 — 10 October 2026

The earlier repository-state paragraphs retain their historical scope. This
branch extends PR4
`de1ba1217d3d78e2790840fcfe5c8b1697056457` with a separate prospective control;
it is not the recovered canonical APEN source or a scientific release.

The legacy bounded bank could change a tick 2 forecast from 12 to 10 after tick 10
ingestion evicted its old record. The new explicit
`apen_event_residual.causal_memory_v2.EpisodicResidualMemoryV2` rejects backward
queries and backdated ingestion, and requires all predictions at a tick to
precede ingestion at that tick. Historical values require replay into fresh
memory. The legacy source and nine original tests remain byte-identical.

Local verification: 14 standard-library unittest cases passed in 0.006 seconds
with exact preserved legacy source; Python 3.12.14, NumPy 2.3.5.
The original nine pytest cases were not run locally because pytest is unavailable.
The safe prototype workflow now runs both original and new tests. Its hosted
acceptance is pending at this initial source commit and is recorded in the draft
PR description after completion. Independent parent source review found no
scoped blocker.

Contract: `research/EVENT_ORDERING_V2_20261010.md`.
Source-bound receipt: `research/verification/event_ordering_v2_20261010/receipt.json`.
All earlier negative conclusions, source-provenance gaps, frozen endpoints and
scientific holds remain. Next action: hosted ordinary engineering verification,
then review this opt-in draft before adoption; no outcome-bearing study follows.
