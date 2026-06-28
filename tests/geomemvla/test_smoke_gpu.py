# tests/geomemvla/test_smoke_gpu.py
# [Geo-MemoryVLA] End-to-end GPU smoke test: geo_only stream (no Qwen3-VL).
# Verifies real VGGT-1B weights + GR00T action head produce a finite loss and
# correct predict_action output shape. Marked @pytest.mark.slow; needs a free GPU.
# Project convention: use GPUs 0-3 for test/smoke runs.
#
# Run:
#   cd /workspace/tingting/starVLA
#   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/tingting/starVLA \
#     HF_HUB_ENABLE_HF_TRANSFER=0 \
#     python -m pytest tests/geomemvla/test_smoke_gpu.py -v -m slow -s
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.slow
def test_forward_and_predict_geo_only():
    """Geo-only smoke: VGGT + GR00T forward + predict, no Qwen3-VL."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    from omegaconf import OmegaConf

    from starVLA.model.framework.base_framework import build_framework

    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")

    # Force geo_only ablation: no Qwen3-VL, no memory, no imagination.
    cfg.framework.world_state.stream = "geo_only"
    cfg.framework.memory.enabled = False
    cfg.framework.imagination.enabled = False

    model = build_framework(cfg).cuda()

    # Confirm the design gap is fixed: no qwen_vl_interface should exist in geo_only mode.
    assert not hasattr(model, "qwen_vl_interface"), (
        "geo_only model must NOT have qwen_vl_interface (Qwen-free ablation violated)"
    )

    # Build 2 fake samples, each with 2 views (224x224), action [16,7], state [1,7].
    rng = np.random.default_rng(0)
    img = Image.fromarray(
        rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    )
    img2 = Image.fromarray(
        rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    )

    def make_sample(lang):
        return {
            "image": [img, img2],  # 2 views -> num_cameras=2
            "lang": lang,
            "action": rng.uniform(-1, 1, (16, 7)).astype(np.float16),
            "state": rng.uniform(-1, 1, (1, 7)).astype(np.float16),
            "episode_id": 0,
            "timestep": 0,
        }

    sample_a = make_sample("pick up the cup")
    sample_b = make_sample("open the drawer")

    # --- forward (training) ---
    out = model([sample_a, sample_b])
    assert "action_loss" in out, f"missing action_loss, got keys: {list(out.keys())}"
    loss = out["action_loss"]
    assert torch.isfinite(loss), f"action_loss is not finite: {loss}"
    print(f"\n[smoke] action_loss = {loss.item():.6f}")

    # --- predict (inference) ---
    pred = model.predict_action([sample_a])
    assert "normalized_actions" in pred, (
        f"missing normalized_actions, got keys: {list(pred.keys())}"
    )
    shape = pred["normalized_actions"].shape
    assert shape == (1, 8, 7), f"expected normalized_actions shape (1,8,7), got {shape}"
    print(f"[smoke] predict_action shape = {shape}  — PASS")
