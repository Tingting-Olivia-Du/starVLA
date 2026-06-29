# tools/d4rt_forecast_probe.py
# [D4RT-WorldState] Phase D-PROBE: offline open-loop extrapolation test. Feeds a clip
# PREFIX to D4RT, queries points at t_tgt beyond the prefix, compares to full-clip GT.
# Pins t_cam = t_tgt (camera-local). No GR00T, no training. See spec §4 / R1 / R2.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

import torch


def build_forecast_query(uv: torch.Tensor, t_src: int, t_tgt: int, device) -> dict:
    """uv: [M,2] in [0,1]. Returns a D4RT query dict with t_cam == t_tgt (spec R2)."""
    M = uv.shape[0]
    u = uv[:, 0].to(device).float().view(1, M)
    v = uv[:, 1].to(device).float().view(1, M)
    tt = torch.full((1, M), int(t_tgt), dtype=torch.long, device=device)
    return {
        "u": u,
        "v": v,
        "t_src": torch.full((1, M), int(t_src), dtype=torch.long, device=device),
        "t_tgt": tt,
        "t_cam": tt.clone(),     # invariant: t_cam == t_tgt
    }


@torch.no_grad()
def probe_extrapolation(model, clip: torch.Tensor, uv: torch.Tensor,
                        prefix_len: int, horizon: int) -> dict:
    """clip: [1,T,3,256,256] in [0,1]. Forecast points at prefix_len..prefix_len+horizon-1
    from a PREFIX-only encode, vs ground truth from the FULL-clip encode.

    Returns {"gt_xyz": [1, M*horizon, 3], "pred_xyz": [1, M*horizon, 3], "mae": float}.
    A low mae means the decoder extrapolates t_tgt > prefix from prefix-only context — the
    R1 forecasting gate. High mae => ship subgoal_type='latent' only.
    """
    device = clip.device
    full_mem = model.encode_video(video=clip, aspect_ratio=None)
    prefix = clip[:, :prefix_len].contiguous()
    pref_mem = model.encode_video(video=prefix, aspect_ratio=None)
    gts, preds = [], []
    for k in range(horizon):
        t = prefix_len + k
        q = build_forecast_query(uv, t_src=0, t_tgt=t, device=device)
        # GT: full clip saw frame t, so this is a reconstruction (the target to match).
        gts.append(model.decode_queries(video=clip, query=q, memory=full_mem)["xyz_3d"])
        # PRED: prefix encode only saw [0, prefix_len); querying t_tgt>=prefix_len is extrapolation.
        preds.append(model.decode_queries(video=prefix, query=q, memory=pref_mem)["xyz_3d"])
    gt = torch.cat(gts, dim=1)
    pred = torch.cat(preds, dim=1)
    return {"gt_xyz": gt, "pred_xyz": pred, "mae": (gt - pred).abs().mean().item()}
