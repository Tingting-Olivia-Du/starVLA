# tests/geomemvla/test_d4rt_config_loads.py
# [D4RT-WorldState] The D4RT config parses and selects the d4rt backbone.
import yaml


def test_d4rt_config_selects_d4rt_backbone():
    with open("examples/LIBERO/train_files/config_geomemvla_d4rt.yaml") as f:
        cfg = yaml.safe_load(f)
    ws = cfg["framework"]["world_state"]
    assert ws["backbone"] == "d4rt"
    assert ws["clip_frames"] == 48
    assert ws["d4rt_model_yaml"].endswith(".yaml")
    assert cfg["framework"]["imagination"]["subgoal_type"] in ("latent", "tracks")
