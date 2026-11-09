"""Full-range inverse-distance arithmetic fixtures, not a scientific benchmark."""
import math
import unittest
from decimal import Decimal, localcontext

import numpy as np

from apen_event_residual.memory import EpisodicResidualMemory
from apen_event_residual.causal_memory_v2 import EpisodicResidualMemoryV2


def decimal_reference(distances, residuals, eps):
    """Independent high-precision reciprocal formula for these fixed fixtures."""
    with localcontext() as context:
        context.prec = 2200
        weights = [Decimal(1) / Decimal.from_float(max(d, eps)) for d in distances]
        numerator = sum(w * Decimal.from_float(r) for w, r in zip(weights, residuals))
        return float(numerator / sum(weights))


def memory_with(cls, distances, residuals, eps):
    memory = cls(key_dim=1, capacity=len(distances), top_k=len(distances),
                 gain=1., eps=eps)
    for distance, residual in zip(distances, residuals):
        memory.observe(key=[distance], residual=residual, available_at=0, observed_at=0)
    return memory


class FullRangeWeightTests(unittest.TestCase):
    def test_tiny_relative_weight_retains_a_normal_contribution(self):
        for cls in (EpisodicResidualMemory, EpisodicResidualMemoryV2):
            for eps, distance, residual in ((1e-300, 1e100, 1e300),
                                             (1e-200, 1e200, 1e200),
                                             (math.ldexp(1., -1000), math.ldexp(1., 500),
                                              math.ldexp(1., 1000))):
                for sign in (-1., 1.):
                    with self.subTest(cls=cls.__name__, eps=eps, sign=sign):
                        residuals = [0., sign * residual]
                        expected = decimal_reference([0., distance], residuals, eps)
                        self.assertNotEqual(expected, 0.)
                        memory = memory_with(cls, [0., distance], residuals, eps)
                        with np.errstate(all='raise'):
                            result = memory.predict(key=[0.], baseline=0., at_time=1)
                        self.assertEqual(result.corrected, expected)
                        self.assertEqual(result.used_entries, 2)

    def test_full_binary64_range_retains_the_smallest_result(self):
        smallest = math.ulp(0.)
        largest = float(np.finfo(float).max)
        for cls in (EpisodicResidualMemory, EpisodicResidualMemoryV2):
            for sign in (-1., 1.):
                with self.subTest(cls=cls.__name__, sign=sign):
                    memory = memory_with(cls, [0., largest], [0., sign * largest], smallest)
                    result = memory.predict(key=[0.], baseline=0., at_time=1)
                    # eps * largest / (largest + eps) rounds to eps.
                    self.assertEqual(result.corrected, sign * smallest)

    def test_far_contribution_is_not_discarded_beside_a_nonzero_neighbor(self):
        smallest = math.ulp(0.)
        distances, residuals = [0., 1e100], [1e-100, 1e308]
        expected = decimal_reference(distances, residuals, smallest)
        self.assertGreater(expected, residuals[0])
        for cls in (EpisodicResidualMemory, EpisodicResidualMemoryV2):
            with self.subTest(cls=cls.__name__):
                result = memory_with(cls, distances, residuals, smallest).predict(
                    key=[0.], baseline=0., at_time=1)
                self.assertEqual(result.corrected, expected)

    def test_cancelling_small_weights_match_reciprocal_reference(self):
        distances = [0., 1e100, 1e100]
        residuals = [1e-100, 1e300, -1e300]
        eps = 1e-300
        expected = decimal_reference(distances, residuals, eps)
        for cls in (EpisodicResidualMemory, EpisodicResidualMemoryV2):
            with self.subTest(cls=cls.__name__):
                memory = memory_with(cls, distances, residuals, eps)
                self.assertEqual(memory.predict(key=[0.], baseline=0., at_time=1).corrected,
                                 expected)

    def test_wide_exponents_match_independent_decimal_and_convex_bounds(self):
        rng = np.random.default_rng(92741)
        for cls in (EpisodicResidualMemory, EpisodicResidualMemoryV2):
            for case in range(24):
                distances = [0.] + [math.ldexp(float(rng.uniform(.5, 1.)), int(e))
                                     for e in rng.integers(-1000, 1001, size=5)]
                residuals = [math.ldexp(float(rng.uniform(-1., 1.)), int(e))
                             for e in rng.integers(-1000, 1001, size=6)]
                eps = math.ldexp(1., -1000)
                expected = decimal_reference(distances, residuals, eps)
                with self.subTest(cls=cls.__name__, case=case):
                    memory = memory_with(cls, distances, residuals, eps)
                    result = memory.predict(key=[0.], baseline=0., at_time=1)
                    self.assertEqual(result.corrected, expected)
                    self.assertGreaterEqual(result.corrected, min(residuals))
                    self.assertLessEqual(result.corrected, max(residuals))


if __name__ == '__main__':
    unittest.main()
