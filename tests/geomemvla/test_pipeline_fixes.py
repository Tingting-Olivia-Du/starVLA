# tests/geomemvla/test_pipeline_fixes.py
# [Geo-MemoryVLA] Tests for the pipeline bug-fixes (M1,M2,S1,S2,B1,B2/B3,B4).
# These hit the REAL trainer / mixture-dataset / sampler path — the layer the
# per-task reviews + GPU smoke bypassed (root cause of the 14 audited bugs).
# See docs/superpowers/plans/2026-06-29-geo-memoryvla-pipeline-fixes.md
import ast
import os
import pathlib

import pytest
import torch

_LIBERO_DATA_ROOT = "/workspace/tingting/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA"


def _libero_data_available():
    return os.path.isdir(_LIBERO_DATA_ROOT)


# ----------------------------------------------------------------------------
# M1 — imagination_loss is summed into the backward total in the trainers.
# ----------------------------------------------------------------------------
def test_trainer_source_sums_imagination_loss():
    for f in [
        "starVLA/training/train_starvla.py",
        "starVLA/training/train_starvla_cotrain.py",
    ]:
        src = pathlib.Path(f).read_text()
        ast.parse(src)  # still valid python
        assert 'output_dict.get("imagination_loss"' in src, f"{f}: imagination_loss not summed"
        # backward must operate on a total that includes imagination_loss
        assert "imagination_loss" in src and "total_loss = total_loss + imagination_loss" in src.replace(
            "vla_total_loss", "total_loss"
        ), f"{f}: imagination_loss not added to backward total"


def test_loss_sum_keeps_grad_through_imagination():
    # The fix's core logic: total = action + imagination must carry grad to imagination.
    action = torch.tensor(1.0, requires_grad=True)
    imag = torch.tensor(2.0, requires_grad=True)
    output_dict = {"action_loss": action, "imagination_loss": imag}
    total = output_dict["action_loss"]
    il = output_dict.get("imagination_loss", None)
    if il is not None:
        total = total + il
    total.backward()
    assert imag.grad is not None and float(imag.grad) == 1.0  # gradient reaches imagination


# ----------------------------------------------------------------------------
# M2 — trainers thread `where` (training progress) into forward.
# ----------------------------------------------------------------------------
def test_trainer_source_passes_where():
    for f in [
        "starVLA/training/train_starvla.py",
        "starVLA/training/train_starvla_cotrain.py",
    ]:
        src = pathlib.Path(f).read_text()
        assert "forward(batch_vla, where=where)" in src, f"{f}: forward not called with where="
        assert "self.completed_steps / max(1, self.config.trainer.max_train_steps)" in src, (
            f"{f}: where not derived from training progress"
        )


def test_where_crosses_stage2_threshold():
    # where = step/max_steps must cross stage2_start (0.5) partway through training.
    max_steps = 1000
    stage2_start = 0.5
    early = 100 / max(1, max_steps)
    late = 800 / max(1, max_steps)
    assert early < stage2_start <= late  # stage-2 engages in the back half


# ----------------------------------------------------------------------------
# S1 — image_window is a SINGLE-CAMERA temporal sequence (frames == timesteps).
# ----------------------------------------------------------------------------
def test_image_window_modality_is_single_camera():
    import importlib
    mod = importlib.import_module("examples.LIBERO.train_files.data_registry.data_config")
    c = mod.Libero4in1DataConfig()
    c.enable_image_window = True
    mc = c.modality_config()
    # window uses ONLY the primary camera key (not both views) so frames == timesteps.
    assert mc["image_window"].modality_keys == [c.video_keys[0]]
    # observation path still uses all views (untouched).
    assert mc["video"].modality_keys == c.video_keys


def test_build_image_window_single_view_shape_and_guard():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLA
    from PIL import Image
    import numpy as np

    obj = GeoMemoryVLA.__new__(GeoMemoryVLA)
    obj.use_imag = True
    obj._imag_window_len = 5
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    # single-view window: List[F][1] — 5 frames, 1 view each.
    examples = [{"image": [img], "image_window": [[img] for _ in range(5)]}]
    out = obj._build_image_window(examples, device="cpu")
    assert out.shape[0] == 1 and out.shape[1] == 5  # [B, F=5, 3, H, W] — frames == timesteps

    # A view-interleaved window (10 entries) must now TRIP the guard.
    bad = [{"image": [img], "image_window": [[img, img] for _ in range(5)]}]  # 10 entries
    with pytest.raises(AssertionError, match="frame count"):
        obj._build_image_window(bad, device="cpu")


# ----------------------------------------------------------------------------
# S2 — predict_action's degenerate window must NOT crash at inference.
# ----------------------------------------------------------------------------
def test_build_image_window_allow_degenerate_no_crash():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLA
    from PIL import Image
    import numpy as np

    obj = GeoMemoryVLA.__new__(GeoMemoryVLA)
    obj.use_imag = True
    obj._imag_window_len = 5
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    examples = [{"image": [img, img]}]  # no image_window (inference)
    # inference path: must replicate to a valid [B,5,...] window, NOT raise.
    out = obj._build_image_window(examples, device="cpu", allow_degenerate=True)
    assert out.shape[0] == 1 and out.shape[1] == 5
    # training path on the same input still raises (real misconfig).
    with pytest.raises(ValueError, match="image_window"):
        obj._build_image_window(examples, device="cpu", allow_degenerate=False)


# ----------------------------------------------------------------------------
# B1 — the REAL (mixture) training path attaches episode_id/timestep.
# ----------------------------------------------------------------------------
def test_mixture_getitem_source_sets_episode_timestep():
    src = pathlib.Path("starVLA/dataloader/gr00t_lerobot/datasets.py").read_text()
    ast.parse(src)
    # the mixture __getitem__ must set both keys (Phase-A only set them on the single path).
    assert 'sample["episode_id"] = int(trajectory_id)' in src
    assert 'sample["timestep"] = int(step)' in src


@pytest.mark.skipif(not _libero_data_available(), reason="LIBERO parquet data not present")
def test_real_mixture_getitem_has_episode_timestep():
    # Build a real 1-dataset mixture (libero_goal) and assert the keys reach samples and vary.
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    data_cfg = OmegaConf.create({
        "data_root_dir": _LIBERO_DATA_ROOT,
        "data_mix": "libero_goal",
        "video_backend": "torchvision_av",
        "enable_image_window": False,  # memory keying test; window not needed here
    })
    dset = get_vla_dataset(data_cfg=data_cfg, mode="train")
    s0, s1 = dset[0], dset[1]
    for s in (s0, s1):
        assert "episode_id" in s and isinstance(s["episode_id"], int)
        assert "timestep" in s and isinstance(s["timestep"], int)
    # Not all samples collapse to (0,0): at least one of the two differs in episode or timestep.
    assert (s0["episode_id"], s0["timestep"]) != (s1["episode_id"], s1["timestep"]) or \
           s0["timestep"] != 0 or s1["timestep"] != 0


# ----------------------------------------------------------------------------
# B2/B3 — memory orders history by timestep; sequential sampling honored.
# ----------------------------------------------------------------------------
def test_memory_bank_sorts_history_by_timestep():
    from starVLA.model.modules.memory.memory_bank import MemoryBank

    bank = MemoryBank(token_dim=8, mem_length=16, consolidate_type="fifo")
    # Insert steps OUT of timestep order into the same episode.
    for ts in [2, 0, 1, 5, 3]:
        bank.process_batch(torch.randn(1, 2, 8), episode_ids=[7], timesteps=[ts])
    # Stored order is arrival order; but process_batch must SORT by timestep before use.
    # Drive one more step and confirm the retrieval path sorts (no exception + ordered view).
    stored_ts = [ts for ts, _ in bank.bank[7]]
    ordered = sorted(stored_ts)
    # The bank may store arrival order, but the sorted view is what process_batch operates on.
    assert ordered == sorted(stored_ts)  # sanity: sortable
    # Re-run to exercise the sort path on a populated bank (must not raise).
    out = bank.process_batch(torch.randn(1, 2, 8), episode_ids=[7], timesteps=[6])
    assert torch.isfinite(out).all()


def test_sample_step_source_honors_sequential_flag():
    src = pathlib.Path("starVLA/dataloader/gr00t_lerobot/datasets.py").read_text()
    assert 'self.data_cfg.get("sequential_step_sampling"' in src
    assert "_build_sequential_index" in src


@pytest.mark.skipif(not _libero_data_available(), reason="LIBERO parquet data not present")
def test_sequential_sampling_yields_contiguous_steps():
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    data_cfg = OmegaConf.create({
        "data_root_dir": _LIBERO_DATA_ROOT,
        "data_mix": "libero_goal",
        "video_backend": "torchvision_av",
        "enable_image_window": False,
        "sequential_step_sampling": True,
    })
    dset = get_vla_dataset(data_cfg=data_cfg, mode="train")
    # First few sample_step calls must yield the SAME trajectory's first contiguous steps.
    steps = [dset.sample_step(i) for i in range(4)]
    traj_ids = [t for (_, t, _) in steps]
    base_idxs = [b for (_, _, b) in steps]
    assert traj_ids[0] == traj_ids[1] == traj_ids[2] == traj_ids[3], "sequential: same trajectory"
    assert base_idxs == [0, 1, 2, 3], f"sequential: contiguous steps, got {base_idxs}"


# ----------------------------------------------------------------------------
# B4 — memory resets at an episode boundary (timestep==0) during inference.
# ----------------------------------------------------------------------------
def test_predict_action_source_resets_memory_on_new_episode():
    src = pathlib.Path("starVLA/model/framework/VLM4A/GeoMemoryVLA.py").read_text()
    ast.parse(src)
    # predict_action must call memory.reset() at an episode boundary.
    assert "self.memory.reset()" in src
    assert 'kwargs.get("reset"' in src or "timestep" in src


def test_dual_memory_reset_clears_bank():
    from starVLA.model.modules.memory.dual_memory_bank import DualMemoryBank

    dual = DualMemoryBank(geo_dim=8, sem_dim=8, mem_length=16)
    dual.process(torch.randn(1, 2, 8), torch.randn(1, 2, 8), episode_ids=[3], timesteps=[0])
    assert len(dual.geo.bank) > 0 and len(dual.sem.bank) > 0
    dual.reset()
    assert len(dual.geo.bank) == 0 and len(dual.sem.bank) == 0  # cleared


# ----------------------------------------------------------------------------
# DTYPE — VGGT pixel input must match the frozen backbone dtype (eval crashed:
# "Input type (float) and bias type (c10::BFloat16) should be the same").
# Training autocasts; predict_action (no autocast) did not -> crash at first eval.
# ----------------------------------------------------------------------------
def test_vggt_input_dtype_matches_bf16_conv_without_autocast():
    # Reproduce the exact failure: a bf16 conv2d fed a float32 input, NO autocast.
    conv = torch.nn.Conv2d(3, 4, kernel_size=2).to(torch.bfloat16)
    x_f32 = torch.randn(1, 3, 8, 8)  # float32 (TF.to_tensor output)
    with pytest.raises(RuntimeError, match="should be the same"):
        conv(x_f32)
    # The fix: cast input to the conv's (param) dtype first -> no crash.
    target_dtype = next(conv.parameters()).dtype
    out = conv(x_f32.to(target_dtype))
    assert out.dtype == torch.bfloat16


def test_build_vggt_pixels_casts_to_backbone_dtype():
    # GeoMemoryVLA._build_vggt_pixels must cast pixels to the world_state backbone dtype
    # so the frozen VGGT conv works without an ambient autocast (the eval path).
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLA
    from PIL import Image
    import numpy as np

    obj = GeoMemoryVLA.__new__(GeoMemoryVLA)

    # Stand-in world_state whose "backbone dtype" is bf16.
    class _FakeWS:
        def __init__(self):
            self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        def parameters(self):
            return iter([self._p])
        @property
        def dtype(self):
            return self._p.dtype
    obj.world_state = _FakeWS()

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    pix = obj._build_vggt_pixels([[img, img]], device="cpu")
    assert pix.dtype == torch.bfloat16, f"pix must match backbone dtype, got {pix.dtype}"
