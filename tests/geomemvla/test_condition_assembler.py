# tests/geomemvla/test_condition_assembler.py
# [Geo-MemoryVLA] Unit test for the multi-stream condition assembler.
import torch
from starVLA.model.modules.geomem.condition_assembler import ConditionAssembler


def test_projects_and_concats_streams():
    asm = ConditionAssembler(stream_dims={"geo": 1024, "sem": 2048}, out_dim=256)
    streams = {"geo": torch.randn(2, 5, 1024), "sem": torch.randn(2, 3, 2048)}
    cond, mask = asm(streams)
    assert cond.shape == (2, 8, 256)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool and mask.all()


def test_skips_none_streams_for_ablation():
    asm = ConditionAssembler(stream_dims={"geo": 1024, "sem": 2048}, out_dim=256)
    streams = {"geo": torch.randn(2, 5, 1024), "sem": None}
    cond, mask = asm(streams)
    assert cond.shape == (2, 5, 256)
