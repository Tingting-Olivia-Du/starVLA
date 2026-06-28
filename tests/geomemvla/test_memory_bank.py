# tests/geomemvla/test_memory_bank.py
# [Geo-MemoryVLA] Unit tests for the lifted MemoryVLA memory bank (dim-agnostic)
# and the dual (geometry + semantic) wrapper.
import torch
from starVLA.model.modules.memory.memory_bank import MemoryBank
from starVLA.model.modules.memory.dual_memory_bank import DualMemoryBank


def test_single_bank_preserves_shape_and_builds_history():
    bank = MemoryBank(token_dim=16, mem_length=4)
    t0 = torch.randn(1, 3, 16)
    out0 = bank.process_batch(t0, episode_ids=[0], timesteps=[0])
    assert out0.shape == (1, 3, 16)
    # Second step in same episode retrieves prior memory (output stays well-formed).
    t1 = torch.randn(1, 3, 16)
    out1 = bank.process_batch(t1, episode_ids=[0], timesteps=[1])
    assert out1.shape == (1, 3, 16)
    assert torch.isfinite(out1).all()


def test_consolidation_caps_memory_length():
    bank = MemoryBank(token_dim=8, mem_length=3, consolidate_type="tome")
    for ts in range(10):
        bank.process_batch(torch.randn(1, 2, 8), episode_ids=[7], timesteps=[ts])
    assert len(bank.bank[7]) <= 3


def test_dual_bank_handles_none_stream():
    dual = DualMemoryBank(geo_dim=16, sem_dim=32, mem_length=4)
    geo = torch.randn(1, 5, 16)
    m_geo, m_sem = dual.process(geo, None, episode_ids=[0], timesteps=[0])
    assert m_geo.shape == (1, 5, 16)
    assert m_sem is None
