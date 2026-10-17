"""Forward-only event ordering for the bounded residual-memory prototype.

Timestamp filtering cannot reconstruct entries evicted by later observations.
This explicit V2 therefore rejects historical queries and backdated ingestions.
It does not change the legacy implementation or establish scientific efficacy.
"""
from __future__ import annotations

from .memory import Correction, EpisodicResidualMemory


class EpisodicResidualMemoryV2(EpisodicResidualMemory):
    """Bounded single-writer memory with prediction-before-ingestion ordering.

    At one clock tick, issue all predictions before ingesting observations for
    that tick. Multiple predictions or observations at the same tick are valid,
    but a prediction after an ingestion at that tick is rejected. Observations
    may contain labels that became available earlier; their ``observed_at`` must
    be the actual current ingestion tick. Failed operations do not advance clocks.

    For historical queries, replay the original chronological events into a new
    instance. No unbounded archive, historical state reconstruction or concurrent
    writer safety is promised by this class.
    """

    def __init__(self, *, key_dim: int, capacity: int = 32, top_k: int = 4,
                 gain: float = 0.5, eps: float = 1e-8):
        super().__init__(key_dim=key_dim, capacity=capacity, top_k=top_k,
                         gain=gain, eps=eps)
        self._last_prediction_at: int | None = None
        self._last_observed_at: int | None = None

    def observe(self, *, key, residual: float, available_at: int, observed_at: int) -> None:
        tick = self._time(observed_at, 'observed_at')
        if self._last_observed_at is not None and tick < self._last_observed_at:
            raise ValueError('observed_at cannot move backwards')
        if self._last_prediction_at is not None and tick < self._last_prediction_at:
            raise ValueError('observed_at cannot predate the last prediction')
        super().observe(key=key, residual=residual,
                        available_at=available_at, observed_at=tick)
        self._last_observed_at = tick

    def predict(self, *, key, baseline: float, at_time: int) -> Correction:
        tick = self._time(at_time, 'at_time')
        if self._last_prediction_at is not None and tick < self._last_prediction_at:
            raise ValueError('at_time cannot move backwards; replay into fresh memory')
        if self._last_observed_at is not None and tick <= self._last_observed_at:
            raise ValueError('prediction must precede ingestion at the same tick; replay into fresh memory')
        result = super().predict(key=key, baseline=baseline, at_time=tick)
        self._last_prediction_at = tick
        return result

    @property
    def last_prediction_at(self) -> int | None:
        return self._last_prediction_at

    @property
    def last_observed_at(self) -> int | None:
        return self._last_observed_at


__all__ = ['EpisodicResidualMemoryV2']
