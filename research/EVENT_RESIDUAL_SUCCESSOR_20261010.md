# Proposed APEN event-residual successor — engineering contract (10 Oct 2026)

**Status: DEVELOPMENT PROTOTYPE ONLY / no scientific results.** The existing APEN and APEN-v2 scientific lines remain **NEGATIVE / NOT SUBMISSION-READY** according to [RESEARCH_STATE.md](../RESEARCH_STATE.md) and [FINAL_STATUS_2026-09-30.md](../FINAL_STATUS_2026-09-30.md). This is a separately named *prospective* successor, not a relabeling of historical APEN v2 or a recovered canonical implementation.

## Narrow engineering objective

Before designing a new architecture/benchmark, implement and inspect one independent causal unit: a fixed-capacity residual memory that can correct an externally supplied prediction using only residual labels that were genuinely observable **and ingested** strictly before the new forecast time. This is a nearest-neighbor engineered control, **not** a new learned APEN architecture, a trained neural model, a simulated Burgers experiment, or a result-bearing comparison.

Interface:
- `observe(key, residual, available_at, observed_at)` refuses observations ingested before their target labels were available, and retains a copied feature key.
- `predict(key, baseline, at_time)` excludes any memory record whose label-availability timestamp **or actual ingestion timestamp** is at/after the queried prediction time.
- Deterministic nearest-context matching, stable tie-breaking, FIFO eviction, bounded capacity and an explicit gain (gain=0 is a no-memory control).
- Return a receipt with eligible/used memory counts and latest retrieved availability time, rather than a spurious uncertainty/performance claim.

### What is not yet decided or authorized

- [ ] Owner/reviewer approval of a new **versioned** hypothesis and any potential scientific publication.
- [ ] Original APEN/legacy source reconciliation and definitive scientific provenance.
- [ ] Selection of an appropriately licensed, representative rare-event dataset and event definition from training/development context only.
- [ ] Train-only normalization and proper target label-availability semantics for that dataset (different time stamps for valid time/observation/ingestion).
- [ ] Fair budget-matched continued-training controls, FNO/operator-learning baselines, model parameter/compute matching and runtime accounting.
- [ ] Predeclared primary metrics, seeds, real independent units, holdout split, OOD regime shifts, stopping rule and anti-peeking gate.
- [ ] Execute outcomes only after a pre-outcome scientific freeze and independent review.

## Verify development-only engineering

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_memory.py
```

**Nine unit tests** were run in a local synthetic scratch environment on 10 October 2026 (all passed). No outcome-bearing scientific experiment or protected data was read. Require hosted exact-head CI and reviewer approval before merging/reusing this code.

This directory must not be cited as a positive APEN result or publication-ready implementation. Old and new evidence shall never be pooled.
