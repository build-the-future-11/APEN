import numpy as np
import pytest
from apen_event_residual.memory import EpisodicResidualMemory


def memory(capacity=2, top_k=1, gain=.5):
    return EpisodicResidualMemory(key_dim=2, capacity=capacity, top_k=top_k, gain=gain)


def test_empty_memory_is_exact_no_memory_control():
    c=memory().predict(key=[0,0],baseline=7.,at_time=1)
    assert c.corrected == 7 and c.used_entries == 0 and c.max_retrieved_available_at is None


def test_reject_residual_before_target_availability():
    m=memory()
    with pytest.raises(ValueError,match='not available'):
        m.observe(key=[0,0],residual=1.,available_at=10,observed_at=9)
    assert m.size == 0


def test_query_excludes_same_time_and_future_records():
    m=memory()
    m.observe(key=[0,0],residual=4.,available_at=10,observed_at=10)
    c=m.predict(key=[0,0],baseline=1.,at_time=10)
    assert c.corrected == 1 and c.eligible_entries == 0
    d=m.predict(key=[0,0],baseline=1.,at_time=11)
    assert d.corrected == 3 and d.max_retrieved_available_at == 10


def test_capacity_fifo_bounded_and_immutable_key():
    m=memory()
    x=np.array([0.,0.])
    m.observe(key=x,residual=100.,available_at=1,observed_at=1)
    x[0] = 99.0
    m.observe(key=[1,1],residual=3.,available_at=2,observed_at=2)
    assert m.predict(key=[0,0],baseline=0,at_time=3).corrected == 50
    m.observe(key=[2,2],residual=9.,available_at=3,observed_at=3)
    assert m.size == 2
    assert m.predict(key=[1,1],baseline=0,at_time=4).corrected == 1.5


def test_deterministic_distance_ties_and_topk():
    m=memory(capacity=3,top_k=2,gain=1.)
    for i,residual in enumerate((2.,4.,20.)):
        m.observe(key=[0,1] if i<2 else [9,9],residual=residual,available_at=i,observed_at=i)
    a=m.predict(key=[0,0],baseline=10,at_time=4)
    b=m.predict(key=[0,0],baseline=10,at_time=4)
    assert a == b
    assert a.used_entries==2 and a.corrected == 13


def test_zero_gain_is_exact_baseline():
    m=memory(gain=0.)
    m.observe(key=[0,0],residual=9.,available_at=0,observed_at=0)
    assert m.predict(key=[0,0],baseline=1.,at_time=1).corrected == 1.


def test_invalid_geometry_and_nonfinite_rejected():
    with pytest.raises(ValueError): EpisodicResidualMemory(key_dim=0)
    with pytest.raises(ValueError): EpisodicResidualMemory(key_dim=2,top_k=5,capacity=2)
    with pytest.raises(ValueError): EpisodicResidualMemory(key_dim=2,gain=1.1)
    with pytest.raises(ValueError): memory().observe(key=[float('nan'),0],residual=1.,available_at=0,observed_at=0)
    with pytest.raises(ValueError): memory().predict(key=[0,0],baseline=float('inf'),at_time=1)


def test_no_future_target_can_be_introduced_by_clock_reversal():
    m=memory(capacity=4,top_k=2,gain=1.)
    m.observe(key=[0,0],residual=1.,available_at=3,observed_at=4)
    m.observe(key=[0,0],residual=100.,available_at=7,observed_at=7)
    assert m.predict(key=[0,0],baseline=0,at_time=7).corrected == 1.
    assert m.predict(key=[0,0],baseline=0,at_time=8).corrected == 50.5


def test_late_ingestion_cannot_change_the_past():
    m = memory()
    m.observe(key=[0,0], residual=100., available_at=2, observed_at=10)
    assert m.predict(key=[0,0], baseline=0, at_time=5).corrected == 0.
    assert m.predict(key=[0,0], baseline=0, at_time=11).corrected == 50.
