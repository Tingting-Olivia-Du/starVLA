# tools/d4rt_probe_libero.py
# [D4RT-WorldState] Phase D-PROBE driver on REAL LIBERO clips. Decides whether the frozen
# D4RT decoder can extrapolate t_tgt>t (the gate for the `tracks` subgoal). Compares D4RT's
# prefix-only extrapolation against (a) a static "no-motion" baseline and (b) D4RT's own
# full-clip reconstruction. No GR00T, no training. See spec §4 / R1.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torchvision.io import read_video

from starVLA.model.modules.d4rt.loader import build_d4rt_model
from tools.d4rt_forecast_probe import build_forecast_query


def load_agentview_clip(video_path: str, num_frames: int, device: str):
    """Read a LIBERO agentview mp4 -> [1, num_frames, 3, 256, 256] in [0,1].
    Uniformly samples num_frames across the episode so the window spans real motion."""
    frames, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")  # [T,3,H,W] uint8
    T = frames.shape[0]
    idx = torch.linspace(0, T - 1, num_frames).round().long()
    clip = frames[idx].float() / 255.0                      # [num_frames,3,H,W]
    # D4RT expects 256x256 (LIBERO already is, but resize defensively).
    if clip.shape[-2:] != (256, 256):
        clip = torch.nn.functional.interpolate(clip, size=(256, 256), mode="bilinear",
                                               align_corners=False)
    return clip.unsqueeze(0).to(device), idx                # [1,num_frames,3,256,256]


@torch.no_grad()
def probe_clip(model, clip, uv, prefix_len, horizon, device):
    """Returns per-frame xyz for: full-clip GT, prefix-only extrapolation, static baseline.
    static baseline = the predicted xyz at the LAST prefix frame, held constant (no motion)."""
    full_mem = model.encode_video(video=clip, aspect_ratio=None)
    prefix = clip[:, :prefix_len].contiguous()
    pref_mem = model.encode_video(video=prefix, aspect_ratio=None)

    # static baseline: where the points are at the last observed prefix frame (per full clip).
    q_last = build_forecast_query(uv, t_src=0, t_tgt=prefix_len - 1, device=device)
    static_xyz = model.decode_queries(video=clip, query=q_last, memory=full_mem)["xyz_3d"]  # [1,M,3]

    gt, pred = [], []
    for k in range(horizon):
        t = prefix_len + k
        q = build_forecast_query(uv, t_src=0, t_tgt=t, device=device)
        gt.append(model.decode_queries(video=clip, query=q, memory=full_mem)["xyz_3d"])
        pred.append(model.decode_queries(video=prefix, query=q, memory=pref_mem)["xyz_3d"])
    gt = torch.stack(gt, dim=1)        # [1, horizon, M, 3]
    pred = torch.stack(pred, dim=1)    # [1, horizon, M, 3]
    static = static_xyz.unsqueeze(1).expand_as(gt)
    return gt, pred, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="playground/Datasets/LEROBOT_LIBERO_DATA/"
                    "libero_object_no_noops_1.0.0_lerobot")
    ap.add_argument("--ckpt", default="playground/Pretrained_models/"
                    "OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt")
    ap.add_argument("--model_yaml", default="starVLA/model/modules/d4rt/model_yaml_48.yaml")
    ap.add_argument("--num_frames", type=int, default=48)   # match the 48CLIP ckpt
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--grid", type=int, default=8)          # 8x8 probe grid
    ap.add_argument("--n_clips", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    model = build_d4rt_model(args.model_yaml, args.ckpt, device=device)
    print(f"[probe] loaded D4RT from {args.ckpt}")

    vids = sorted(glob.glob(os.path.join(
        args.dataset, "videos", "**", "observation.images.image", "*.mp4"), recursive=True))
    vids = vids[: args.n_clips]
    print(f"[probe] {len(vids)} agentview clips, num_frames={args.num_frames}, "
          f"prefix={args.num_frames - args.horizon}, horizon={args.horizon}")

    g = args.grid
    ys, xs = torch.meshgrid(torch.linspace(0.15, 0.85, g), torch.linspace(0.15, 0.85, g),
                            indexing="ij")
    uv = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)  # central grid

    prefix_len = args.num_frames - args.horizon
    d4rt_maes, static_maes, recon_maes = [], [], []
    for i, vp in enumerate(vids):
        clip, _ = load_agentview_clip(vp, args.num_frames, device)
        gt, pred, static = probe_clip(model, clip, uv, prefix_len, args.horizon, device)
        # reconstruction sanity: full-clip query at the SAME target frames should match GT (==GT
        # here by construction, so instead measure full vs prefix encode AT an OBSERVED frame).
        q_obs = build_forecast_query(uv, t_src=0, t_tgt=prefix_len - 1, device=device)
        full_mem = model.encode_video(video=clip, aspect_ratio=None)
        pref_mem = model.encode_video(video=clip[:, :prefix_len].contiguous(), aspect_ratio=None)
        recon = (model.decode_queries(video=clip, query=q_obs, memory=full_mem)["xyz_3d"]
                 - model.decode_queries(video=clip[:, :prefix_len], query=q_obs, memory=pref_mem)["xyz_3d"]
                 ).abs().mean().item()

        d4rt_maes.append((gt - pred).abs().mean().item())
        static_maes.append((gt - static).abs().mean().item())
        recon_maes.append(recon)
        if i < 5:
            print(f"  clip {i:02d}: d4rt_extrap={d4rt_maes[-1]:.4f}  "
                  f"static={static_maes[-1]:.4f}  recon_gap={recon:.4f}")

    d, s, r = np.array(d4rt_maes), np.array(static_maes), np.array(recon_maes)
    print("\n================ D-PROBE RESULT ================")
    print(f"clips                : {len(d)}")
    print(f"D4RT extrapolation MAE: {d.mean():.4f} ± {d.std():.4f}")
    print(f"Static (no-motion) MAE: {s.mean():.4f} ± {s.std():.4f}")
    print(f"Recon gap (obs frame) : {r.mean():.4f} ± {r.std():.4f}   (should be small)")
    skill = (s.mean() - d.mean()) / max(s.mean(), 1e-6)
    print(f"\nMotion skill score    : {skill:+.3f}   "
          f"(>0 = beats static; want clearly >0)")
    verdict = ("PASS — D4RT extrapolates better than static; tracks is VIABLE"
               if skill > 0.10 else
               "FAIL — D4RT no better than static; ship subgoal_type=latent ONLY")
    print(f"GATE                  : {verdict}")
    print("===============================================")


if __name__ == "__main__":
    main()
