# tests/geomemvla/test_geomemvla_framework.py
# [Geo-MemoryVLA] Verifies the framework registers and its default config carries
# the ablation switches. Heavy model construction is covered by the slow smoke test.
def test_framework_is_registered():
    import starVLA.model.framework.base_framework as bf
    bf._auto_import_framework_modules()
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    assert "GeoMemoryVLA" in FRAMEWORK_REGISTRY._registry


def test_default_config_has_ablation_switches():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLADefaultConfig
    cfg = GeoMemoryVLADefaultConfig()
    assert "enabled" in cfg.memory
    assert "enabled" in cfg.imagination
    assert cfg.world_state["stream"] in ("geo_only", "sem_only", "dual")
