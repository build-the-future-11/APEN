"""Causally gated bounded episodic residual correction (development prototype).

This class stores *only* residual observations whose labels have become
available. Queries cannot read records with availability timestamps greater
than or equal to the prediction time. It contains no neural predictor and has
not been evaluated on any scientific dataset or protected benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Correction:
    corrected: float
    baseline: float
    residual_correction: float
    eligible_entries: int
    used_entries: int
    max_retrieved_available_at: int | None


@dataclass(frozen=True)
class _Record:
    key: np.ndarray
    residual: float
    available_at: int
    observed_at: int
    insertion: int


class EpisodicResidualMemory:
    """A fixed-capacity, nearest-context residual correction control.

    **Not an efficacy claim.** Context normalization, event definition,
    equal-budget training and dataset identity require separate preregistration.
    No learned gate, attention optimizer or trained JEPA is implemented.
    """

    def __init__(self, *, key_dim: int, capacity: int = 32, top_k: int = 4,
                 gain: float = 0.5, eps: float = 1e-8):
        if type(key_dim) is not int or key_dim < 1:
            raise ValueError('key_dim must be positive integer')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive integer')
        if type(top_k) is not int or not 1 <= top_k <= capacity:
            raise ValueError('top_k must be in [1, capacity]')
        if not np.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError('gain must be finite in [0, 1]')
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')
        self.key_dim, self.capacity, self.top_k = key_dim, capacity, top_k
        self.gain, self.eps = float(gain), float(eps)
        self._records: list[_Record] = []
        self._insertions = 0

    def _key(self, key) -> np.ndarray:
        arr = np.asarray(key, dtype=np.float64)
        if arr.shape != (self.key_dim,) or not np.isfinite(arr).all():
            raise ValueError(f'key must be finite with shape ({self.key_dim},)')
        return arr.copy()

    @staticmethod
    def _time(value: int, name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be nonnegative integer clock tick')
        return value

    def observe(self, *, key, residual: float, available_at: int, observed_at: int) -> None:
        """Insert only after the actual residual's target becomes observable."""
        a = self._time(available_at, 'available_at')
        b = self._time(observed_at, 'observed_at')
        if a > b:
            raise ValueError('target residual not available when observed')
        if not np.isfinite(residual):
            raise ValueError('residual must be finite')
        record = _Record(self._key(key), float(residual), a, b, self._insertions)
        self._insertions += 1
        self._records.append(record)
        # Capacity and eviction policy are fixed; never examine prediction error.
        if len(self._records) > self.capacity:
            self._records.pop(0)

    def predict(self, *, key, baseline: float, at_time: int) -> Correction:
        """Use only recorded residuals strictly available before query time."""
        t = self._time(at_time, 'at_time')
        query = self._key(key)
        if not np.isfinite(baseline):
            raise ValueError('baseline must be finite')
        eligible = [r for r in self._records if r.available_at < t and r.observed_at < t]
        if not eligible:
            return Correction(float(baseline), float(baseline), 0.0, 0, 0, None)
        # Deterministic nearest-neighbor selection, including equal-distance ties.
        distances = [(float(np.linalg.norm(query - r.key)), r.available_at, r.insertion, r)
                     for r in eligible]
        if not all(np.isfinite(d) for d, _, _, _ in distances):
            raise ValueError('nonfinite context distance')
        distances.sort(key=lambda x: (x[0], x[1], x[2]))
        chosen = distances[:self.top_k]
        weights = np.asarray([1.0 / max(d, self.eps) for d, _, _, _ in chosen], dtype=float)
        weights /= weights.sum()
        learned_residual = float(weights @ np.asarray([r.residual for _, _, _, r in chosen]))
        amount = self.gain * learned_residual
        prediction = float(baseline) + amount
        if not np.isfinite(prediction):
            raise ValueError('nonfinite corrected prediction')
        return Correction(prediction, float(baseline), amount, len(eligible), len(chosen),
                          max(r.available_at for _, _, _, r in chosen))

    @property
    def size(self) -> int:
        return len(self._records)
