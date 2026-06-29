# tests/geomemvla/test_world_state_factory.py
# [D4RT-WorldState] Factory dispatches on world_state.backbone (no model download).
import pytest


def test_factory_defaults_to_vggt_world_class():
    from starVLA.model.modules.geomem.world_state_factory import _world_state_cls
    from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
    assert _world_state_cls("vggt_world") is WorldStateAdapter


def test_factory_selects_d4rt_class():
    from starVLA.model.modules.geomem.world_state_factory import _world_state_cls
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
    assert _world_state_cls("d4rt") is D4RTWorldStateAdapter


def test_factory_rejects_unknown_backbone():
    from starVLA.model.modules.geomem.world_state_factory import _world_state_cls
    with pytest.raises(ValueError):
        _world_state_cls("nope")
