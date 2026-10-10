"""Artificial chronological stream regressions; no scientific data or outcomes."""
import unittest

from apen_event_residual.causal_memory_v2 import EpisodicResidualMemoryV2
from apen_event_residual.memory import EpisodicResidualMemory


def _memory(cls=EpisodicResidualMemoryV2, capacity=1):
    return cls(key_dim=1, capacity=capacity, top_k=1, gain=1.)


def _observe(memory, tick, residual=2., available=None):
    memory.observe(key=[0.], residual=residual,
                   available_at=tick if available is None else available,
                   observed_at=tick)


def _predict(memory, tick):
    return memory.predict(key=[0.], baseline=10., at_time=tick)


class ChronologicalMemoryTests(unittest.TestCase):
    def test_legacy_witness_future_fifo_eviction_changes_past_prediction(self):
        memory = _memory(EpisodicResidualMemory)
        _observe(memory, 1)
        before = _predict(memory, 2)
        _observe(memory, 10, residual=100.)
        after = _predict(memory, 2)
        self.assertEqual(before.corrected, 12.)
        self.assertEqual(after.corrected, 10.)

    def test_v2_rejects_historical_query_after_future_fifo_eviction(self):
        memory = _memory()
        _observe(memory, 1)
        self.assertEqual(_predict(memory, 2).corrected, 12.)
        _observe(memory, 10, residual=100.)
        with self.assertRaisesRegex(ValueError, 'replay into fresh memory'):
            _predict(memory, 2)
        self.assertEqual(_predict(memory, 11).corrected, 110.)
        self.assertEqual(memory.size, 1)

    def test_prediction_clock_cannot_reverse_without_observations(self):
        memory = _memory()
        self.assertEqual(_predict(memory, 8).corrected, 10.)
        with self.assertRaisesRegex(ValueError, 'move backwards'):
            _predict(memory, 7)
        self.assertEqual(memory.last_prediction_at, 8)

    def test_same_tick_forecasts_precede_ingestion_even_when_fifo_evicts(self):
        memory = _memory()
        _observe(memory, 1)
        self.assertEqual(_predict(memory, 2).corrected, 12.)
        _observe(memory, 2, residual=4.)
        with self.assertRaisesRegex(ValueError, 'precede ingestion'):
            _predict(memory, 2)
        self.assertEqual(_predict(memory, 3).corrected, 14.)

    def test_backdated_ingestion_cannot_enter_an_already_queried_past(self):
        memory = _memory()
        _predict(memory, 10)
        with self.assertRaisesRegex(ValueError, 'predate the last prediction'):
            _observe(memory, 9)
        self.assertEqual(memory.size, 0)
        self.assertIsNone(memory.last_observed_at)
        _observe(memory, 10, available=2)
        self.assertEqual(_predict(memory, 11).corrected, 12.)

    def test_observation_clock_cannot_reverse(self):
        memory = _memory()
        _observe(memory, 4)
        with self.assertRaisesRegex(ValueError, 'move backwards'):
            _observe(memory, 3)
        self.assertEqual(memory.last_observed_at, 4)
        self.assertEqual(memory.size, 1)

    def test_multiple_forecasts_before_same_tick_ingestion_are_allowed(self):
        memory = _memory()
        _observe(memory, 1)
        first = _predict(memory, 3)
        self.assertEqual(first, _predict(memory, 3))
        _observe(memory, 3, residual=9.)
        self.assertEqual(_predict(memory, 4).corrected, 19.)

    def test_multiple_observations_at_one_tick_preserve_fifo_policy(self):
        memory = _memory()
        _predict(memory, 4)
        _observe(memory, 4, residual=2.)
        _observe(memory, 4, residual=3.)
        result = _predict(memory, 5)
        self.assertEqual(result.corrected, 13.)
        self.assertEqual(result.eligible_entries, 1)
        self.assertEqual(result.max_retrieved_available_at, 4)

    def test_failed_observation_does_not_advance_clocks_or_evict(self):
        memory = _memory()
        _observe(memory, 1)
        with self.assertRaises(ValueError):
            memory.observe(key=[float('nan')], residual=5., available_at=10, observed_at=10)
        self.assertEqual(memory.last_observed_at, 1)
        self.assertEqual(memory.size, 1)
        self.assertEqual(_predict(memory, 2).corrected, 12.)

    def test_failed_prediction_does_not_advance_clock(self):
        memory = _memory()
        with self.assertRaises(ValueError):
            memory.predict(key=[0.], baseline=float('nan'), at_time=10)
        self.assertIsNone(memory.last_prediction_at)
        _observe(memory, 1)
        self.assertEqual(_predict(memory, 2).corrected, 12.)

    def test_timestamp_admission_rejects_unavailable_target_without_clock_change(self):
        memory = _memory()
        with self.assertRaisesRegex(ValueError, 'not available'):
            _observe(memory, 4, available=5)
        self.assertIsNone(memory.last_observed_at)
        self.assertEqual(_predict(memory, 0).corrected, 10.)

    def test_tick_zero_has_prediction_then_observation_order(self):
        memory = _memory()
        self.assertEqual(_predict(memory, 0).corrected, 10.)
        _observe(memory, 0)
        with self.assertRaisesRegex(ValueError, 'precede ingestion'):
            _predict(memory, 0)
        self.assertEqual(_predict(memory, 1).corrected, 12.)

    def test_forward_stream_predictions_match_legacy_receipts_exactly(self):
        legacy, current = _memory(EpisodicResidualMemory, capacity=3), _memory(capacity=3)
        for tick in range(12):
            self.assertEqual(_predict(legacy, tick), _predict(current, tick))
            _observe(legacy, tick, residual=float(tick), available=max(0, tick - 2))
            _observe(current, tick, residual=float(tick), available=max(0, tick - 2))
            self.assertLessEqual(current.size, 3)

    def test_historical_prediction_is_reproducible_by_fresh_chronological_replay(self):
        live = _memory()
        _observe(live, 1)
        historical = _predict(live, 2)
        _observe(live, 10, residual=100.)
        replay = _memory()
        _observe(replay, 1)
        self.assertEqual(historical, _predict(replay, 2))


if __name__ == '__main__':
    unittest.main()
