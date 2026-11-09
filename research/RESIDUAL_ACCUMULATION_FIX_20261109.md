# Finite residual-mean arithmetic repair: 9 November 2026

**Current implementation:** the reviewed final correction uses exact rational
accumulation of the already-rounded float weights and residuals, followed by one
float conversion. The initial scaled implementation below is preserved as a
failed intermediate approach; see the review correction at the end of this file.

## Defect and impact

The prospective V1/V2 residual-memory prototype rejects a representable finite
weighted mean when the direct float64 dot product overflows. At source
`86f7d6c0cbf34c3f767f3271dcb4e953ac32d092`, eleven identical neighbours with
residual `float64.max`, baseline zero and gain one produce an arithmetic failure
instead of the same finite maximum value. The same failure occurs with gain zero,
even though that forecast should equal its baseline.

Direct weighted accumulation also loses the unit mean in the equal-weight
sequence `[float64.max, 3, -float64.max]`, and rounds eleven identical smallest
positive subnormal residuals to zero before summation. These are tiny artificial
engineering counterexamples; no historical APEN outcome is implicated.

## Change

Retain the existing relative inverse-distance weights before normalization. Scale
each residual by the largest absolute residual, sum the weighted scaled values
with `math.fsum`, divide by the accurately summed weights, and restore the common
residual scale. All-zero residuals return a zero correction without division.
Each scaled numerator term has magnitude no greater than its weight, so the
weighted mean is formed within the residual scale before the final multiplication.

Distance, eligibility, FIFO eviction, top-k selection, availability/insertion tie
priority, gain, receipt fields and V2 event clocks are unchanged. Genuine overflow
of baseline plus correction is still rejected, without advancing the V2 clock.
Ordinary last-bit results can differ because summation order and normalization
change. This is floating-point arithmetic, not exact real arithmetic: extreme
relative weights or scaled residuals can still round to zero.

## Verification

```sh
python3 -m pytest -q -W error tests/test_memory.py tests/test_memory_v2.py
```

Python 3.12.14 and NumPy 2.3.5: **37 tests passed, 72 subtests passed**, zero failures
or skips. The six added methods cover identical maximum residuals with several
neighbour counts/signs/gains, unequal weights, cancellation, zero/subnormal
values, an independent 80-digit Decimal ordinary-value oracle, and genuine final
prediction overflow with clock preservation. Existing distance, weight and
chronology regressions remain in the full suite.

Running just the new class on the unchanged predecessor source produced
**46 failed test/subtest cases**, with four reported passes and two passing
subtests. Raw before/after process logs and source identities are retained in
`research/verification/residual_accumulation_20261109/`.

The change is stacked on the existing inventory correction in PR #8, which is
itself stacked on the context-distance repair in PR #6. Their source and review
histories remain separate. The new commit uses `[skip ci]`; final-head hosted
verification and maintainer review remain open. No experiment, protected-data
access, training, manuscript revision, submission, merge or release is included.

## Scientific boundary

This code remains a prospective engineering prototype. The original spatial-PDE
source recovery and negative comparison are unchanged. Passing these arithmetic
tests does not establish predictive efficacy or make the original research
release complete.

## Independent review correction: exact accumulation

Two separate AI agent reviews found that global residual scaling can erase a
representable small contribution. With equal weights, the residual sequence
`[1e308, -1e308, 1e-100]` became zero instead of approximately `3.33e-101`.
A zero-weight outlier with residual `1e308` could also set the scale and erase a
nearest-neighbour contribution of `1e-100`. The same risk persists with positive
subnormal weights when the extreme terms cancel. These findings block the
initial scaled implementation; its commit, process logs and initial receipt are
retained rather than overwritten.

The final implementation represents each nonzero admitted float weight and
finite stored residual with `fractions.Fraction`, accumulates their products and
the weight sum exactly, divides, then converts the mean to float once. At least
one relative weight is exactly one, so the denominator is positive. The exact
mean lies within the finite residual extrema and therefore has a finite rounded
float result. Zero-weight entries do not contribute. The final baseline-plus-
gain correction still has the original finite-output check.

This precision applies to accumulation of the **already-computed float weights**.
The earlier distance and relative-weight divisions remain floating-point
operations; their rounding and possible underflow are unchanged. Exact rational
arithmetic adds CPU work per retained entry; runtime throughput has not been
benchmarked. No predictive-performance statement follows from this repair.

Three added review-driven methods cover the equal-weight dynamic-range
cancellation, the zero-weight outlier, and the positive-subnormal-weight
cancellation. The first two methods produce five failed test/subtest cases on
the initial scaled implementation. Final full-suite result: **40 tests and
78 subtests pass**, warnings treated as errors. The operations agent separately
read the corrected source and reran the full suite, obtaining the same pass
counts; its review found no remaining blocker in the admitted finite-float
scope. This is AI code review, not human scientific review.

Latest receipt: `research/verification/residual_accumulation_20261109/followup_receipt.json`.
Final-head hosted verification and maintainer integration remain open.
