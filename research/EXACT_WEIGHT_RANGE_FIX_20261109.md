# Preserve contributions from very small inverse-distance weights

## Problem

The prospective residual-memory implementation at commit
`59bd5dfbe96980d3f4f365b604fec2b88b8caf6a` uses exact accumulation after forming
relative weights with floating-point division. That division can lose a useful
contribution before the exact accumulator sees it.

Consider a query at zero, epsilon `1e-300`, and two admitted records:

| Context distance | Residual |
| ---: | ---: |
| 0 | 0 |
| 1e100 | 1e300 |

The parent rounds the second relative weight, approximately `1e-400`, to zero.
The resulting correction is zero. The inverse-distance weighted mean is
approximately `1e-100`, which is representable. A second witness spans the
binary64 range: epsilon is the smallest positive subnormal, and the far
distance and far residual are both the largest finite float. The correct mean
rounds to that smallest subnormal; the parent instead returns zero.

## Change and mathematical boundary

For each selected record, let `d_i = max(context_distance_i, eps)`. Every
`d_i` has already been checked finite and is strictly positive. The candidate
uses the exact rational value `w_i = 1 / d_i`, forms
`sum(w_i * residual_i) / sum(w_i)` exactly, then rounds that mean to float once.
All weights are positive, so the denominator is positive and the mean lies
between the smallest and largest selected finite residuals. Its conversion
therefore stays finite.

The change removes intermediate reciprocal/relative-weight overflow,
underflow and rounding. It does not undo rounding in the Euclidean distance,
gain multiplication or final baseline addition. Genuine final prediction
overflow still raises and V2 clocks are not advanced by failed predictions.

Eligibility, FIFO retention, nearest-neighbor ordering, ties, capacity, top_k,
gain, and the event-order contract are untouched. Ordinary results may differ
by an ulp because weights are no longer rounded before accumulation. The old
zero-rounded-weight test was updated to the full mathematical expectation:
the very distant record contributes a small, representable increment.

## Verification

Run the complete suite from the repository root:

```sh
python3 -m pytest -q -W error tests/
```

The parent fails 29 new test/subtest checks in `test_weight_range.py`.
The repaired complete suite passes **45 tests and 146 subtests**, with warnings
treated as errors. New fixtures include positive/negative normal results,
the smallest representable result, cancellation, nonzero near-neighbor
contributions, and wide exponent ranges compared with an independent
2,200-digit Decimal reciprocal formula. Existing causal-order, retrieval,
overflow, FIFO and failure-atomicity tests remain in the suite.

A separate AI reviewer independently reran the suite, reviewed the finite
convex-mean argument, and checked 100 additional seeded cases for each class
against a Decimal reference. All 200 class evaluations matched. This is an
engineering review, not human scientific peer review.

The exact provider version of the parent module has Git blob
`5b7b2b5efc55c214834456e7f04c4d105b039f04`. Raw logs, candidate hashes,
the reviewer receipt and timing samples are retained in the adjacent
`verification/exact_weights_20261109` directory.

## Measured local CPU cost

The timing script compares identical artificial banks and queries, alternates
parent/candidate order over seven batches, and records all samples. Recover
the reference module from the recorded parent commit and run:

```sh
git show 59bd5dfbe96980d3f4f365b604fec2b88b8caf6a:apen_event_residual/memory.py > /tmp/apen-reference-memory.py
PYTHONPATH=. python3 research/benchmark_weight_range.py \
  --baseline-source /tmp/apen-reference-memory.py \
  --baseline-commit 59bd5dfbe96980d3f4f365b604fec2b88b8caf6a \
  --output /tmp/apen-weight-range-timing.json
```

| Capacity | top_k | Parent median (microseconds) | Candidate median (microseconds) | Candidate / parent |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 107.7 | 114.7 | 1.07 |
| 32 | 32 | 322.7 | 320.2 | 0.99 |
| 128 | 128 | 1218.8 | 2416.0 | 1.98 |
| 256 | 32 | 802.1 | 832.3 | 1.04 |

These are descriptive results from one local run, not a robust speed ranking.
Timer noise can reverse small differences. Exact rational denominators can
grow as top_k increases; the large-bank case shows material overhead. No
upper-bound latency or target-hardware performance is established. The script
records Python/NumPy versions, platform, seed, source hashes and all batches.

## Integration and scientific status

This is an isolated draft stacked on PR #10. The commit uses `[skip ci]`;
hosted CI has not run. A maintainer must review and integrate it separately.
No scientific data, training, frozen endpoint, paper, release or production
configuration changed. The original APEN negative scientific result remains
unchanged, and these arithmetic fixtures establish no forecasting advantage.
