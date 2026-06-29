# [D4RT-WorldState] Thin loader: model.yaml + .ckpt -> D4RTModel (strict=False).
# Mirrors Open-D4RT eval_track3d_in_worldtrack.py:313-325.
# Vendored-API caller — see github.com/Lijiaxin0111/Open-d4rt
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

from starVLA.model.modules.d4rt.core.config import load_yaml_config
from starVLA.model.modules.d4rt.core.checkpoint import load_checkpoint
from starVLA.model.modules.d4rt.model.builder import build_model


def _unwrap_state_dict(payload):
    # Mirrors Open-D4RT _unwrap_state_dict: unwrap nested state_dict/model/module/network/net.
    if isinstance(payload, dict):
        for k in ("state_dict", "model", "module", "network", "net"):
            if k in payload and isinstance(payload[k], dict):
                return payload[k]
    return payload


def build_d4rt_model(model_yaml: str, ckpt_path: str | None = None, device: str = "cpu"):
    cfg = load_yaml_config(model_yaml)
    model = build_model(cfg["model"]).eval()
    if ckpt_path is not None:
        sd = _unwrap_state_dict(load_checkpoint(ckpt_path, map_location="cpu"))
        if not sd:
            raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval()
