# tests/geomemvla/test_world_state_adapter.py
# [Geo-MemoryVLA] Adapter import + GeometryState flatten contract (no model download).
import torch


def test_geometry_state_flatten_contract():
    # Exercises the vendored GeometryState shape contract without loading VGGT.
    from starVLA.model.modules.vggt_world.state import GeometryState

    b, f, tok, d = 2, 3, 7, 1024
    st = GeometryState(
        frame_tokens=torch.randn(b, f, tok, d),
        global_tokens=torch.randn(b, f, tok, d),
        patch_start_idx=5,
        patch_grid=(1, 1),
    )
    flat = st.flatten_streams()
    assert flat.shape == (b, f * 2 * tok, d)


def test_adapter_imports_and_exposes_hidden_size():
    from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
    # Class is importable and declares the dim without instantiating VGGT.
    assert WorldStateAdapter.HIDDEN_SIZE == 1024
