# tools/d4rt_probe_visualize.py
# [D4RT-WorldState] D-PROBE visualization. Produces:
#   (1) agentview frames with predicted (red) vs ground-truth (green) query points overlaid,
#       so the decoder's drift is visible on the image plane.
#   (2) per-horizon MAE bar chart: D4RT extrapolation vs static baseline vs GT motion magnitude.
# Frozen D4RT, real LIBERO clips. No GR00T, no training. See spec §4 / R1 / D-PROBE gate.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import read_video

from starVLA.model.modules.d4rt.loader import build_d4rt_model
from tools.d4rt_forecast_probe import build_forecast_query

OUT = "visualization/d4rt_probe"


def load_contiguous_clip(video_path, num_frames, device):
    """Contiguous window from the middle of the episode (arm actively moving) -> [1,F,3,256,256]."""
    fr, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")  # [T,3,H,W] uint8
    T = fr.shape[0]
    start = max(0, T // 2 - num_frames // 2)
    clip = fr[start:start + num_frames]
    if clip.shape[0] < num_frames:                      # pad short episodes
        clip = torch.cat([clip, clip[-1:].repeat(num_frames - clip.shape[0], 1, 1, 1)], 0)
    clip01 = (clip.float() / 255.0).unsqueeze(0).to(device)
    return clip01, clip  # also return uint8 frames for plotting


@torch.no_grad()
def query_uv_xyz(model, video, memory, uv, t_tgt, device):
    q = build_forecast_query(uv, t_src=0, t_tgt=t_tgt, device=device)
    out = model.decode_queries(video=video, query=q, memory=memory)
    return out["uv_2d"][0], out["xyz_3d"][0]            # [M,2] in [0,1], [M,3]


def viz_overlay(model, clip01, frames_u8, uv, prefix_len, horizon, device, tag):
    """Plot GT (green) vs prefix-extrapolated (red) query points on the target agentview frames."""
    full = model.encode_video(video=clip01, aspect_ratio=None)
    pref = model.encode_video(video=clip01[:, :prefix_len].contiguous(), aspect_ratio=None)
    H, W = frames_u8.shape[-2:]

    fig, axes = plt.subplots(1, horizon, figsize=(5 * horizon, 5), squeeze=False)
    for k in range(horizon):
        t = prefix_len + k
        gt_uv, _ = query_uv_xyz(model, clip01, full, uv, t, device)        # GT (full clip saw t)
        pr_uv, _ = query_uv_xyz(model, clip01[:, :prefix_len], pref, uv, t, device)  # extrapolated
        gt = gt_uv.cpu().numpy() * np.array([W, H])
        pr = pr_uv.cpu().numpy() * np.array([W, H])
        ax = axes[0][k]
        ax.imshow(frames_u8[t].permute(1, 2, 0).numpy())
        ax.scatter(gt[:, 0], gt[:, 1], c="lime", s=28, marker="o", edgecolors="black",
                   linewidths=0.4, label="GT (full clip)")
        ax.scatter(pr[:, 0], pr[:, 1], c="red", s=28, marker="x",
                   label="D4RT prefix-extrap")
        for j in range(gt.shape[0]):                                       # error vector
            ax.plot([gt[j, 0], pr[j, 0]], [gt[j, 1], pr[j, 1]], c="yellow", lw=0.6, alpha=0.7)
        err = np.abs(gt - pr).mean()
        ax.set_title(f"t_tgt={t} (h={k+1})  2D MAE={err:.1f}px")
        ax.axis("off")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"D4RT forecast vs GT — {tag}  (red drifts from green ⇒ no extrapolation)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f"overlay_{tag}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def collect_mae_by_horizon(model, clips, uv, prefix_len, horizon, device):
    """Per-horizon MAE for D4RT extrapolation, static baseline, and GT motion magnitude."""
    d4 = np.zeros(horizon); st = np.zeros(horizon); mo = np.zeros(horizon); n = 0
    for clip01, _ in clips:
        full = model.encode_video(video=clip01, aspect_ratio=None)
        pref = model.encode_video(video=clip01[:, :prefix_len].contiguous(), aspect_ratio=None)
        _, static_xyz = query_uv_xyz(model, clip01, full, uv, prefix_len - 1, device)  # last prefix
        for k in range(horizon):
            t = prefix_len + k
            _, gt_xyz = query_uv_xyz(model, clip01, full, uv, t, device)
            _, pr_xyz = query_uv_xyz(model, clip01[:, :prefix_len], pref, uv, t, device)
            d4[k] += (gt_xyz - pr_xyz).abs().mean().item()
            st[k] += (gt_xyz - static_xyz).abs().mean().item()
            mo[k] += (gt_xyz - static_xyz).abs().mean().item()  # GT displacement = motion magnitude
        n += 1
    return d4 / n, st / n, mo / n, n


def viz_bars(d4, st, mo, n, horizon):
    x = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(max(7, horizon * 0.9), 5))
    w = 0.35
    ax.bar(x - w / 2, d4, w, label="D4RT extrapolation MAE", color="#d62728")
    ax.bar(x + w / 2, st, w, label="Static (no-motion) MAE", color="#7f7f7f")
    ax.plot(x, mo, "o--", c="#2ca02c", label="GT motion magnitude")
    ax.set_xlabel("forecast horizon (frames ahead of prefix)")
    ax.set_ylabel("MAE (normalized camera units)")
    skill = (st.mean() - d4.mean()) / max(st.mean(), 1e-6)
    ax.set_title(f"D-PROBE: D4RT extrapolation vs baselines  ({n} clips)\n"
                 f"motion-skill = {skill:+.2f}  ⇒  "
                 f"{'PASS' if skill > 0.10 else 'FAIL → latent-only'}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "mae_by_horizon.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="playground/Datasets/LEROBOT_LIBERO_DATA/"
                    "libero_object_no_noops_1.0.0_lerobot")
    ap.add_argument("--ckpt", default="playground/Pretrained_models/"
                    "OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt")
    ap.add_argument("--model_yaml", default="starVLA/model/modules/d4rt/model_yaml_48.yaml")
    ap.add_argument("--num_frames", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--n_clips", type=int, default=15)
    ap.add_argument("--n_overlay", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = args.device
    model = build_d4rt_model(args.model_yaml, args.ckpt, device=device)
    print(f"[viz] loaded D4RT, out dir = {OUT}")

    g = args.grid
    ys, xs = torch.meshgrid(torch.linspace(0.15, 0.85, g), torch.linspace(0.15, 0.85, g),
                            indexing="ij")
    uv = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)
    prefix_len = args.num_frames - args.horizon

    vids = sorted(glob.glob(os.path.join(
        args.dataset, "videos", "**", "observation.images.image", "*.mp4"), recursive=True))
    vids = vids[: args.n_clips]

    clips = [load_contiguous_clip(vp, args.num_frames, device) for vp in vids]

    # (1) overlays for the first n_overlay clips
    saved = []
    for i in range(min(args.n_overlay, len(clips))):
        clip01, frames_u8 = clips[i]
        p = viz_overlay(model, clip01, frames_u8, uv, prefix_len, args.horizon, device,
                        tag=f"clip{i:02d}")
        saved.append(p)
        print(f"[viz] saved {p}")

    # (2) MAE-by-horizon bar chart over all clips
    d4, st, mo, n = collect_mae_by_horizon(model, clips, uv, prefix_len, args.horizon, device)
    p = viz_bars(d4, st, mo, n, args.horizon)
    saved.append(p)
    print(f"[viz] saved {p}")
    print("\n[viz] DONE. files:")
    for s in saved:
        print("  ", s)


if __name__ == "__main__":
    main()
