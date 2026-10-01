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
