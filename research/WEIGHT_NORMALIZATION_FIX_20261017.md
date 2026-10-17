# Inverse-distance normalization repair — 17 October 2026

Package 0.1.1 corrects the shared numerical normalization used by the V1 and V2
development prototypes. The mathematical weighting rule and V2 chronological
admission rules are unchanged. This is a source correction, not a new APEN
efficacy result or a rerun of any historical scientific study.

## Reproduced defect

Parent commit `aa61af7d221d7a4da2d8ee032bc7c3ec0592d41a` passed all 23 existing
unit tests. With two zero-distance neighbours, residuals 2 and 4, baseline 10,
gain 1 and epsilon `1e-308`, its prediction was 10 instead of 13: summing the
two finite reciprocal weights overflowed and normalization set both to zero.
All 14 reconstructed parent files were checked against their Git blob IDs
before reproduction.

## Correction and compatibility

For denominators `a_i = max(distance_i, epsilon)`, form `min(a) / a_i` and
normalize those relative weights. The cancelled common factor does not affect
the intended weighted mean. Every intermediate weight is at most one and one
is exactly one, avoiding reciprocal and reciprocal-sum overflow. Relative
weights too small for float64 may round to zero; underflow is ignored only in
that division. All other floating-point error handling is preserved.

The default and chronological classes both receive this correction. Their
ordinary results can differ by floating-point rounding. The FIFO policy,
distance calculation, top-k selection, gain, receipts and chronology are
unchanged. V1 still does not support reliable historical queries after eviction;
use V2 and replay historical events into a fresh instance.

Earlier committed verification receipts remain tied to their recorded source
and are not rewritten as evidence for package 0.1.1. Existing negative research
conclusions and protected experiment gates remain unchanged.

## Verification

On Python 3.12.14 and NumPy 2.3.5:

```sh
python -m pytest -q -W error tests/test_memory.py tests/test_memory_v2.py
```

Result: **26 tests passed; 8 subtests passed**. This includes all 23 original
cases plus three new test methods: equal neighbours over the epsilon range for
both classes, an independent Decimal inverse-distance mean with unequal
distances, and an extreme relative-weight case. The regression calculations
also run with NumPy floating-point errors set to raise. The original witness
now produces 13. These are small synthetic engineering fixtures.

The unchanged workflow includes both existing test files and will verify the
published source separately. No hosted success is asserted by this local note.
