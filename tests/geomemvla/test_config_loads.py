# [Geo-MemoryVLA] The training config parses and selects the right framework.
from omegaconf import OmegaConf


def test_config_parses_and_names_framework():
    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    assert cfg.framework.name == "GeoMemoryVLA"
    assert cfg.framework.world_state.stream in ("geo_only", "sem_only", "dual")
    assert cfg.framework.action_model.action_dim == 7
