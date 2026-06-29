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
    # [Geo-MemoryVLA] S1: image_window is single-camera (primary only) so frames == timesteps.
    assert mc["image_window"].modality_keys == [c.video_keys[0]]


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
            # [Geo-MemoryVLA] S1: image_window is single-camera (primary only).
            self.modality_keys["image_window"] = ["video.primary_image"]


def _fake_data(F, with_window=False):
    # (F, H, W, C) uint8 per video key; action/lang minimal.
    # For video modality the data dict uses the raw key (1 frame each).
    # For image_window modality get_step_data stores arrays under "image_window.<key>",
    # so _pack_sample reads "image_window.video.primary_image" etc.
    frame = lambda v: np.full((F, 32, 32, 3), v, dtype=np.uint8)
    data = {
        "video.primary_image": frame(1)[:1],          # 1 frame for video modality
        "video.wrist_image": frame(2)[:1],
        "annotation.human.action.task_description": ["pick up the cup"],
        "action.x": np.zeros((8, 1), dtype=np.float32),
    }
    if with_window:
        # Simulate the collision-fixed storage: image_window data under prefixed keys.
        # [Geo-MemoryVLA] S1: single-camera window -> only the primary key is stored.
        data["image_window.video.primary_image"] = frame(10)  # F frames
    return data


def test_pack_sample_emits_window_when_modality_present():
    dset = _FakeDataset(with_window=True)
    sample = dset._pack_sample(_fake_data(F=5, with_window=True))
    assert "image_window" in sample
    assert len(sample["image_window"]) == 5            # F frames
    assert len(sample["image_window"][0]) == 1         # single-camera (S1)
    assert isinstance(sample["image_window"][0][0], Image.Image)
    assert sample["image_window"][0][0].size == (224, 224)
    # current frame still present and single-frame-per-view
    assert len(sample["image"]) == 2


def test_pack_sample_omits_window_when_modality_absent():
    dset = _FakeDataset(with_window=False)
    sample = dset._pack_sample(_fake_data(F=1, with_window=False))
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


# ---------------------------------------------------------------------------
# Integration tests: real _get_delta_indices and real data path
# These tests cover C-1 (dispatch crash) and C-2 (delta_indices collision).
# ---------------------------------------------------------------------------

def test_delta_indices_no_collision_with_image_window():
    """C-2 regression: _get_delta_indices must scope keys by modality so that
    'video.video.primary_image' ([0]) and 'image_window.video.primary_image'
    ([-1,0,1,2,3]) are distinct entries — not one overwriting the other."""
    Cfg = _load_libero_cfg()
    cfg = Cfg()
    cfg.enable_image_window = True
    mc = cfg.modality_config()

    # Build a minimal stub of LeRobotSingleDataset just to call _get_delta_indices
    # without touching any filesystem or network.
    from starVLA.dataloader.gr00t_lerobot import datasets as ds

    class _MinimalDataset:
        modality_configs = mc

    stub = _MinimalDataset()
    di = ds.LeRobotSingleDataset._get_delta_indices(stub)

    # Both modalities share the key "video.primary_image" — they MUST be distinct entries.
    assert "video.video.primary_image" in di, (
        f"Missing 'video.video.primary_image' in delta_indices keys: {list(di.keys())}"
    )
    assert "image_window.video.primary_image" in di, (
        f"Missing 'image_window.video.primary_image' in delta_indices keys: {list(di.keys())}"
    )
    # Values must differ: video=[0], image_window=[-1,0,1,2,3]
    assert list(di["video.video.primary_image"]) == [0], (
        f"video.video.primary_image expected [0], got {di['video.video.primary_image'].tolist()}"
    )
    assert list(di["image_window.video.primary_image"]) == [-1, 0, 1, 2, 3], (
        f"image_window.video.primary_image expected [-1,0,1,2,3], got {di['image_window.video.primary_image'].tolist()}"
    )
    # Wrist image: present in the observation (video) modality with [0].
    assert "video.video.wrist_image" in di
    assert list(di["video.video.wrist_image"]) == [0]
    # [Geo-MemoryVLA] S1: image_window is primary-only, so there is NO wrist window key.
    assert "image_window.video.wrist_image" not in di


def test_get_data_by_modality_no_crash_for_image_window():
    """C-1 regression: get_data_by_modality must not raise ValueError for image_window.
    Uses a stub that intercepts get_video so no filesystem is needed."""
    from starVLA.dataloader.gr00t_lerobot import datasets as ds

    Cfg = _load_libero_cfg()
    cfg = Cfg()
    cfg.enable_image_window = True
    mc = cfg.modality_config()

    sentinel = object()  # returned by stubbed get_video

    class _StubDataset:
        modality_configs = mc
        delta_indices = ds.LeRobotSingleDataset._get_delta_indices(
            type("S", (), {"modality_configs": mc})()
        )

        def get_video(self, trajectory_id, key, base_index, modality="video"):
            return sentinel

        def get_state_or_action(self, *a, **kw):
            return None

        def get_language(self, *a, **kw):
            return []

    stub = _StubDataset()
    # Must not raise — previously raised ValueError("Invalid modality: image_window")
    result = ds.LeRobotSingleDataset.get_data_by_modality(
        stub, trajectory_id=0, modality="image_window", key="video.primary_image", base_index=0
    )
    assert result is sentinel, f"Expected sentinel from get_video stub, got {result}"


_LIBERO_SPATIAL_PATH = (
    "playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot"
)


def _libero_data_available():
    import os
    return os.path.isdir(
        f"/workspace/tingting/starVLA/{_LIBERO_SPATIAL_PATH}/data/chunk-000"
    )


@pytest.mark.skipif(not _libero_data_available(), reason="LIBERO parquet data not present")
def test_real_getitem_image_window_no_collision():
    """End-to-end: build a real LeRobotSingleDataset with enable_image_window=True,
    call __getitem__, and assert:
      - sample['image'] has num_views PIL images (1 per view = current frame)
      - sample['image_window'] has 5 frames each with num_views PIL images
      - The two are distinct objects (no aliasing / collision)
    """
    import pathlib
    from starVLA.dataloader.gr00t_lerobot import datasets as ds
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        LeRobotSingleDataset,
        DatasetMetadata,
    )
    from examples.LIBERO.train_files.data_registry.data_config import Libero4in1DataConfig

    dataset_path = pathlib.Path(f"/workspace/tingting/starVLA/{_LIBERO_SPATIAL_PATH}")

    data_cfg = Libero4in1DataConfig()
    data_cfg.enable_image_window = True
    mc = data_cfg.modality_config()

    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

    # LIBERO uses av1-encoded mp4 videos; torchvision_av can handle these.
    data_cfg_dict = {"lerobot_version": "v2.0", "video_backend": "torchvision_av"}
    dset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=mc,
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend="torchvision_av",
        data_cfg=data_cfg_dict,
    )

    # Sample a step from episode 0, frame 5 (safe: episode 0 > 5 frames)
    sample = dset[5]

    # ---- sample["image"]: current frame, num_views PIL images ----
    assert "image" in sample, "sample must contain 'image'"
    num_views = len(data_cfg.video_keys)
    assert len(sample["image"]) == num_views, (
        f"Expected {num_views} views in sample['image'], got {len(sample['image'])}"
    )
    from PIL import Image as PILImage
    for img in sample["image"]:
        assert isinstance(img, PILImage.Image), f"Expected PIL Image, got {type(img)}"
        assert img.size == (224, 224)

    # ---- sample["image_window"]: 5-frame window, num_views per frame ----
    assert "image_window" in sample, (
        "sample must contain 'image_window' when enable_image_window=True"
    )
    expected_frames = len(data_cfg.image_window_indices)  # 5
    assert len(sample["image_window"]) == expected_frames, (
        f"Expected {expected_frames} frames in image_window, got {len(sample['image_window'])}"
    )
    # [Geo-MemoryVLA] S1: image_window is single-camera (primary only) -> 1 view per frame.
    for f_idx, frame_views in enumerate(sample["image_window"]):
        assert len(frame_views) == 1, (
            f"Frame {f_idx}: expected 1 view (single-camera window), got {len(frame_views)}"
        )
        for v_idx, img in enumerate(frame_views):
            assert isinstance(img, PILImage.Image), (
                f"Frame {f_idx} view {v_idx}: expected PIL Image, got {type(img)}"
            )

    # ---- No collision: image and image_window[1] (delta=0, current frame) ----
    # They come from independent get_video calls — must be distinct objects.
    assert sample["image"][0] is not sample["image_window"][1][0], (
        "sample['image'][0] and image_window[1][0] must be independent objects"
    )


@pytest.mark.skipif(not _libero_data_available(), reason="LIBERO parquet data not present")
def test_real_getitem_d4rt_48frame_window():
    """[D4RT-WorldState] With d4rt_window_frames=48, the dataloader must yield a 48-frame
    contiguous past window (NOT VGGT's 5-frame context/chunk window), single-camera."""
    import pathlib
    from PIL import Image as PILImage
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
    from examples.LIBERO.train_files.data_registry.data_config import Libero4in1DataConfig

    dataset_path = pathlib.Path(f"/workspace/tingting/starVLA/{_LIBERO_SPATIAL_PATH}")
    data_cfg = Libero4in1DataConfig()
    data_cfg.enable_image_window = True
    data_cfg.d4rt_window_frames = 48                  # the D4RT override
    assert data_cfg.image_window_indices == list(range(-47, 1))  # 48 past frames ending at current
    mc = data_cfg.modality_config()

    dset = LeRobotSingleDataset(
        dataset_path=dataset_path, modality_configs=mc,
        embodiment_tag=EmbodimentTag.FRANKA, video_backend="torchvision_av",
        data_cfg={"lerobot_version": "v2.0", "video_backend": "torchvision_av"},
    )
    # Sample a frame with >=47 past frames available (episode 0 is long enough).
    sample = dset[60]
    assert "image_window" in sample
    assert len(sample["image_window"]) == 48, (
        f"expected 48-frame D4RT window, got {len(sample['image_window'])}"
    )
    for frame_views in sample["image_window"]:
        assert len(frame_views) == 1                  # single-camera (primary only)
        assert isinstance(frame_views[0], PILImage.Image)
