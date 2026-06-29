# tests/geomemvla/test_d4rt_smoke_gpu.py
# [D4RT-WorldState] End-to-end Phase D0: GeoMemoryVLA(backbone=d4rt), geo_only stream
# (no Qwen3-VL), imagination OFF. Verifies the vendored Open-D4RT encoder + GR00T action
# head produce a finite loss and correct predict_action shape. Marked @pytest.mark.slow.
# Mirrors tests/geomemvla/test_smoke_gpu.py. Project convention: GPUs 4-7 for smoke runs.
#
# Run (needs the 14GB OpenD4RT ckpt placed per config_geomemvla_d4rt.yaml):
#   cd /workspace/tingting/starVLA
#   CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/workspace/tingting/starVLA \
#     HF_HUB_ENABLE_HF_TRANSFER=0 \
#     python -m pytest tests/geomemvla/test_d4rt_smoke_gpu.py -v -m slow -s
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.slow
def test_d4rt_geo_only_forward_and_predict():
    """Phase D0: D4RT encoder + GR00T forward + predict, geo_only, imagination OFF."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")

    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework

    cfg = OmegaConf.load("examples/LIBERO/train_files/config_geomemvla_d4rt.yaml")
    # Phase D0 ablation: D4RT encoder only, no Qwen3-VL, no memory, no imagination.
    cfg.framework.world_state.stream = "geo_only"
    cfg.framework.memory.enabled = False
    cfg.framework.imagination.enabled = False

    model = build_framework(cfg).cuda()
    assert model._backbone == "d4rt", "expected the D4RT backbone to be selected"
    assert not hasattr(model, "qwen_vl_interface"), "geo_only must NOT build Qwen3-VL"

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    img2 = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))

    def make_sample(lang):
        return {
            "image": [img, img2],   # 2 views; D4RT takes only view 0 (agentview, B1)
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
    assert "action_loss" in out, f"missing action_loss, got: {list(out.keys())}"
    assert torch.isfinite(out["action_loss"]), f"action_loss not finite: {out['action_loss']}"
    print(f"\n[d4rt-smoke] action_loss = {out['action_loss'].item():.6f}")

    # --- predict (inference) ---
    pred = model.predict_action([sample_a])
    assert "normalized_actions" in pred, f"missing normalized_actions: {list(pred.keys())}"
    shape = tuple(pred["normalized_actions"].shape)
    assert shape == (1, 8, 7), f"expected (1,8,7), got {shape}"
    print(f"[d4rt-smoke] predict_action shape = {shape}  — PASS")
