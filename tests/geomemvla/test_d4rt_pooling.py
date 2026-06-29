# tests/geomemvla/test_d4rt_pooling.py
# [D4RT-WorldState] geo/imag condition pooling: D4RT memory is [B, T*256(+1), C] — pooling
# compresses the 256 spatial patches per time-step to k, keeping the temporal axis, so the
# GR00T cross-attention condition stays bounded (was 6145 tokens @ 48 frames -> OOM).
import torch


def test_pool_spatial_keeps_temporal_structure():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import pool_d4rt_tokens
    B, C = 2, 1280
    T_steps, S = 24, 256                     # 48 frames -> 24 temporal steps, 256 spatial patches
    mem = torch.randn(B, T_steps * S + 1, C)  # +1 leading global/aspect token
    out = pool_d4rt_tokens(mem, k=8, spatial=S)
    # 24 steps * 8 pooled + 1 leading = 193 tokens, dim unchanged.
    assert out.shape == (B, T_steps * 8 + 1, C)


def test_pool_no_leading_token():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import pool_d4rt_tokens
    B, C = 1, 64
    T_steps, S = 4, 256
    mem = torch.randn(B, T_steps * S, C)     # exact multiple, no leading token
    out = pool_d4rt_tokens(mem, k=8, spatial=S)
    assert out.shape == (B, T_steps * 8, C)


def test_pool_passthrough_when_already_short():
    # If N is already small (<= target), don't pool — return as-is (no spurious reshape).
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import pool_d4rt_tokens
    mem = torch.randn(2, 10, 64)
    out = pool_d4rt_tokens(mem, k=8, spatial=256)
    assert out.shape == (2, 10, 64)


def test_adapter_flatten_pools_when_configured():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTState
    B, C = 2, 1280
    st = D4RTState(memory=torch.randn(B, 24 * 256 + 1, C), video=torch.randn(B, 48, 3, 256, 256))
    flat = st.flatten(pool_k=8)
    assert flat.shape == (B, 24 * 8 + 1, C)
    # default (no pooling) is identity
    assert st.flatten().shape == (B, 24 * 256 + 1, C)
