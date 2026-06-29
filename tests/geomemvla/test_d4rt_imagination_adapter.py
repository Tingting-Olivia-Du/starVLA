# tests/geomemvla/test_d4rt_imagination_adapter.py
# [D4RT-WorldState] Adapter import + subgoal-head shape contract (no model download).
import torch


def test_adapter_imports_both_subgoal_types():
    from starVLA.model.modules.geomem.d4rt_imagination_adapter import D4RTImaginationAdapter
    assert hasattr(D4RTImaginationAdapter, "imagine_tokens")
    assert hasattr(D4RTImaginationAdapter, "training_loss")


def test_latent_head_projects_memory_to_subgoal():
    # The 'latent' forecaster head maps [B,N,C] -> [B,N,C] without a D4RT model.
    from starVLA.model.modules.geomem.d4rt_imagination_adapter import LatentForecastHead
    head = LatentForecastHead(dim=64, horizon=2)
    mem = torch.randn(2, 10, 64)
    out = head(mem)
    assert out.shape == (2, 10, 64)
