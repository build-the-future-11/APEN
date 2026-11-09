# APEN — RESEARCH_STATE

## Research question
Does episodic residual retrieval improve a spatial forecaster beyond strong matched alternatives under a valid causal protocol?

## Current scientific status
The current APEN line should be treated as a **NEGATIVE** result with respect to the strong superiority claim.

The retained manuscript history contains small improvements over a frozen base, but the strongest equal-update control favored continued base training. The current repository status further records that the repaired/current APEN comparison is not submission-ready and that reduced validation-selected comparisons do not support superiority.

## Repository/release state
Historical snapshot (superseded by the 7 November 2026 inventory): the public default branch formerly contained only `README.md` and `FINAL_STATUS_2026-09-30.md`.

The older README referenced paths including `APEN.md`, `STATUS.md`, `RESEARCH_TRUTH.md`, `REPRODUCE.md`, `scripts/preflight.py`, and `paper/apen_v2/README.md`. Those historical paths remain unverified as a complete spatial-PDE scientific release. The separate `apen_event_residual/`, `tests/`, and `research/` materials are a prospective engineering prototype, not recovery of that paper's source/evidence. See `research/REPOSITORY_BOUNDARY_20261107.md`.

Therefore the public repository is **not a reproducible release of the original spatial-PDE APEN experiments**, despite the separately documented prototype and tests.

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

### Engineering arithmetic correction — 17 October 2026

Package 0.1.1 repairs inverse-distance weight normalization in the shared V1/V2
prototype implementation. See
[`research/WEIGHT_NORMALIZATION_FIX_20261017.md`](research/WEIGHT_NORMALIZATION_FIX_20261017.md)
for the reproduced failure, compatibility boundary, and 26-test local acceptance.
The earlier V1/V2 source-preservation and verification statements describe their
recorded October 10 commits. They do not assert byte equality with this later
arithmetic correction. Historical scientific evidence and conclusions remain
unchanged; no outcome-bearing study is opened.

### Context-distance arithmetic correction — 5 November 2026

The prospective prototype now uses a scaled Euclidean distance to prevent
squaring underflow from turning distinct neighbours into false ties and
squaring overflow from rejecting representable distances. The README now
describes the prototype files already present instead of saying that the
repository contains only two status files. This does not restore the historical
forecasting implementation.

The source correction and independent regression evidence are documented in
[`research/DISTANCE_SCALING_FIX_20261105.md`](research/DISTANCE_SCALING_FIX_20261105.md).
The entire local engineering suite passed: 31 tests and 26 subtests. Existing
scientific conclusions, evidence records and experiment holds are unchanged.

### Residual-mean arithmetic correction: 9 November 2026

A further shared V1/V2 correction scales finite residuals before accurate
weighted summation. This prevents direct-dot overflow of representable convex
means and preserves the tested cancellation/subnormal cases. It leaves causal
admission, retrieval, gains and the historical APEN evidence boundary intact.

The unchanged predecessor fails 46 new test/subtest cases. The repaired full
local suite passes 37 tests and 72 subtests with warnings treated as errors.
Exact source identities, raw logs, commands and remaining gates are in
`research/verification/residual_accumulation_20261109/receipt.json`; rationale
and limitations are in `research/RESIDUAL_ACCUMULATION_FIX_20261109.md`.
This is a stacked draft based on PR #8, not a merged or released correction.
Final-head hosted verification and maintainer review remain pending. No
scientific checkpoint is advanced and no submitted manuscript is changed.
