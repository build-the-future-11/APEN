# Preserve context-distance ordering across float64 scales

This corrects the prospective residual-memory prototype at parent commit
`7bd34e86f0c100bd8eb8ee767a2ad8f90e243d7f`. It does not evaluate APEN on a
scientific dataset or alter its historical negative result.

## Reproduced failure

The shared V1/V2 implementation calculates distance with
`np.linalg.norm(query - key)`. Squaring coordinates directly can underflow to
zero or overflow even when the Euclidean distance itself is representable.

An artificial two-record bank has capacity 2, top-k 1, gain 1 and baseline 10.
The query is `[0, 0]`. Insert the farther key `[0, 2s]` with residual 90 before
the nearer key `[s, 0]` with residual 3. The intended prediction is 13 for every
positive scale `s` in this example.

At the parent revision, both classes return 100 for `s = 1e-200`: both distances
become zero and insertion order breaks the false tie. For `s = 1e200`, both
classes raise `ValueError: nonfinite context distance` after overflow. At
ordinary scale `s = 1`, both return 13.

## Correction

Use the standard library's scaled `math.dist` calculation. Finite very large
distances and tiny nonzero separations retain their ordering. Distances whose
true magnitude exceeds float64's range still reach the existing nonfinite
distance rejection. FIFO capacity, causal filtering, distance/availability/
insertion tie priority, inverse-distance weighting and gains are unchanged.

Ordinary distance values may differ by floating-point rounding. This is not a
bitwise-compatibility promise for near-tied geometries. The intended metric
remains Euclidean distance. The same correction applies to V1 and V2; V1's
historical-query limitation is unchanged.

## Verification

Local Python 3.12.14, NumPy 2.3.5 and pytest 9.2.0:

```sh
python -m pytest -q -W error tests/test_memory.py tests/test_memory_v2.py
```

Result: **31 tests passed, 26 subtests passed**. All 26 previous test methods
remain, plus five methods covering:

- Nearest-neighbour identity at scales `1e-200`, `1` and `1e200`.
- Multidimensional inverse-distance means checked against an independent
  100-digit Decimal calculation on the exact represented float inputs.
- Distinct subnormal distances, including the smallest positive float64 value.
- True ties, with availability priority followed by insertion priority.
- Genuine distance overflow rejected without changing V2's prediction clock.

Copying only those new tests to a detached checkout of the unchanged parent
produces **16 failing subcases**. The ordinary-scale controls pass. The tests
set NumPy floating-point errors to raise, so a warning cannot silently hide a
range failure.

These are small implementation fixtures. No training, protected-data access,
historical outcome replay, benchmark campaign, manuscript change, scientific
submission or production release occurred. Independent source review and
hosted verification remain separate from these local results.
