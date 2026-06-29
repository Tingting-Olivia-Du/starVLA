# tests/geomemvla/test_d4rt_world_state_adapter.py
# [D4RT-WorldState] Adapter import + D4RTState flatten contract (no model download).
import torch


def test_d4rt_state_flatten_contract():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTState
    b, n, c = 2, 17, 512
    st = D4RTState(memory=torch.randn(b, n, c), video=torch.randn(b, 4, 3, 256, 256))
    flat = st.flatten()
    assert flat.shape == (b, n, c)


def test_adapter_imports():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
    assert hasattr(D4RTWorldStateAdapter, "encode")
    assert hasattr(D4RTWorldStateAdapter, "flatten")


def test_hidden_size_read_from_model_at_construction():
    # hidden_size must be known immediately (from heads.xyz_head.in_features), so the
    # orchestrator's geo_dim read at construction works without a forward pass.
    import glob
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
    yamls = glob.glob("starVLA/model/modules/d4rt/**/model_yaml*.yaml", recursive=True)
    assert yamls, "vendored model.yaml missing"
    adapter = D4RTWorldStateAdapter(model_yaml=yamls[0], ckpt_path=None)
    # 48CLIP model.yaml has decoder.hidden_dim: 1280.
    assert adapter.hidden_size == 1280
