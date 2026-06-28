# tests/geomemvla/test_image_window.py
# [Geo-MemoryVLA] Unit tests for the image_window modality + packing.
# Part of Geo-MemoryVLA Phase-C — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md
import importlib


def _load_libero_cfg():
    mod = importlib.import_module("examples.LIBERO.train_files.data_registry.data_config")
    return mod.Libero4in1DataConfig


def test_image_window_indices_derivation():
    Cfg = _load_libero_cfg()
    c = Cfg()
    # paper k=2, m=2 -> [-1,0,1,2,3], length == context+chunk+1 == 5
    assert c.image_window_indices == [-1, 0, 1, 2, 3]
    assert len(c.image_window_indices) == c.image_window_context + c.image_window_chunk + 1


def test_modality_config_includes_window_when_enabled():
    Cfg = _load_libero_cfg()
    c = Cfg()
    c.enable_image_window = True
    mc = c.modality_config()
    assert "image_window" in mc
    assert mc["image_window"].delta_indices == [-1, 0, 1, 2, 3]
    assert mc["image_window"].modality_keys == c.video_keys


def test_modality_config_omits_window_when_disabled():
    Cfg = _load_libero_cfg()
    c = Cfg()
    c.enable_image_window = False
    mc = c.modality_config()
    assert "image_window" not in mc
    # observation path untouched
    assert mc["video"].delta_indices == [0]
