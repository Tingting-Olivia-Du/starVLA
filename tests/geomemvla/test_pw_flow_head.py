# [LangPointWorld] Point-flow head shapes + gradients (CPU).
import torch
from starVLA.model.modules.langpw.point_flow_head import PointFlowHead


def test_shapes_and_grad():
    B, L, H = 2, 10, 256
    head = PointFlowHead(cond_dim=H, n_points=64, flow_horizon=4, hidden=128)
    cond = torch.randn(B, L, H, requires_grad=True)
    out = head(cond)
    assert out["flow"].shape == (B, 64, 4, 3)
    assert out["flow_tokens"].shape == (B, 64, 128)
    out["flow"].sum().backward()
    assert cond.grad is not None


def test_mask_is_accepted():
    B, L, H = 2, 10, 256
    head = PointFlowHead(cond_dim=H, n_points=32, flow_horizon=4, hidden=128)
    cond = torch.randn(B, L, H)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, 5:] = False
    out = head(cond, encoder_attention_mask=mask)
    assert out["flow"].shape == (B, 32, 4, 3)
