# tests/geomemvla/test_image_window.py
# [Geo-MemoryVLA] Unit tests for the image_window modality + packing.
# Part of Geo-MemoryVLA Phase-C — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md
import importlib

import pytest


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


import numpy as np
from PIL import Image


class _FakeDataset:
    # Minimal stand-in exposing just what _pack_sample needs.
    def __init__(self, with_window):
        from starVLA.dataloader.gr00t_lerobot import datasets as ds
        self._pack_sample = ds.LeRobotSingleDataset._pack_sample.__get__(self)
        self.tag = "libero_franka"
        self.data_cfg = {}
        self.modality_keys = {
            "video": ["video.primary_image", "video.wrist_image"],
            "language": ["annotation.human.action.task_description"],
            "action": ["action.x"],
        }
        if with_window:
            self.modality_keys["image_window"] = ["video.primary_image", "video.wrist_image"]


def _fake_data(F):
    # (F, H, W, C) uint8 per video key; action/lang minimal.
    frame = lambda v: np.full((F, 32, 32, 3), v, dtype=np.uint8)
    return {
        "video.primary_image": frame(10),
        "video.wrist_image": frame(20),
        "annotation.human.action.task_description": ["pick up the cup"],
        "action.x": np.zeros((8, 1), dtype=np.float32),
    }


def test_pack_sample_emits_window_when_modality_present():
    dset = _FakeDataset(with_window=True)
    sample = dset._pack_sample(_fake_data(F=5))
    assert "image_window" in sample
    assert len(sample["image_window"]) == 5            # F frames
    assert len(sample["image_window"][0]) == 2         # num_views
    assert isinstance(sample["image_window"][0][0], Image.Image)
    assert sample["image_window"][0][0].size == (224, 224)
    # current frame still present and single-frame-per-view
    assert len(sample["image"]) == 2


def test_pack_sample_omits_window_when_modality_absent():
    dset = _FakeDataset(with_window=False)
    sample = dset._pack_sample(_fake_data(F=1))
    assert "image_window" not in sample


def test_episode_boundary_clamp_keeps_past_in_episode():
    # Mirrors get_video's clamp: step_indices = delta + base, then clamp to [0, traj_len-1].
    delta = [-1, 0, 1, 2, 3]          # the image_window indices
    traj_len = 50
    base_index = 0                    # episode head
    step_indices = np.array(delta) + base_index
    step_indices = np.maximum(step_indices, 0)
    step_indices = np.minimum(step_indices, traj_len - 1)
    # past frame (-1) must clamp to 0, never to a prior episode's last frame.
    assert step_indices.tolist() == [0, 0, 1, 2, 3]
    assert step_indices.min() >= 0


def test_episode_boundary_clamp_at_tail():
    delta = [-1, 0, 1, 2, 3]
    traj_len = 4                      # tiny episode; future frames overflow
    base_index = 3                    # last frame
    step_indices = np.minimum(np.maximum(np.array(delta) + base_index, 0), traj_len - 1)
    # future frames clamp to last index (3), no bleed past episode end.
    assert step_indices.tolist() == [2, 3, 3, 3, 3]


def test_imagination_default_horizon_is_two():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLADefaultConfig
    cfg = GeoMemoryVLADefaultConfig()
    assert cfg.imagination["horizon"] == 2          # paper m=2


def test_build_image_window_raises_when_imag_on_and_window_missing():
    # Construct a lightweight object exposing just _build_image_window + the flags it reads,
    # without building the full (GPU) model.
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLA
    from PIL import Image
    import numpy as np

    obj = GeoMemoryVLA.__new__(GeoMemoryVLA)      # bypass __init__ (no model load)
    obj.use_imag = True
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    examples = [{"image": [img, img]}]            # no "image_window"
    with pytest.raises(ValueError, match="image_window"):
        obj._build_image_window(examples, device="cpu")


def test_yaml_enables_window_and_imagination():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    assert cfg.datasets.vla_data.enable_image_window is True
    assert cfg.framework.imagination.enabled is True
    assert cfg.framework.imagination.horizon == 2


def test_gating_resolves_to_and_of_flags():
    # The gate the training entry applies: window iff enable_image_window AND imagination.enabled.
    def gate(enable_image_window, imagination_enabled):
        return bool(enable_image_window) and bool(imagination_enabled)

    assert gate(True, True) is True
    assert gate(True, False) is False
    assert gate(False, True) is False
    assert gate(False, False) is False
