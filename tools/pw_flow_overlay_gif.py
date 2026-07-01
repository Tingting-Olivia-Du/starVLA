# [LangPointWorld] Overlay the teacher's imagined point-flow on the LIBERO agentview and save a
# GIF + frames: predicted flow arrows (projected world->pixel) on the real demo RGB frames sampled
# at the same full-demo timesteps. Lets you eyeball predicted motion vs actual object motion
# (and the under-drive) directly. Reads the cached imagine sample (no model reload).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse
import glob
import os

import numpy as np
import cv2
import h5py
import imageio.v2 as imageio


def project(points_world, K, E_world_to_cam):
    """points_world [N,3] -> pixel [N,2] (float), and depth [N]. E maps world->cam."""
    ones = np.ones((points_world.shape[0], 1), points_world.dtype)
    cam = (E_world_to_cam @ np.concatenate([points_world, ones], -1).T).T[:, :3]  # [N,3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def draw_flow(img, uv0, uv1, mask, color, thickness=1):
    out = img.copy()
    for (a, b, m) in zip(uv0, uv1, mask):
        if not m:
            continue
        p0 = (int(round(a[0])), int(round(a[1])))
        p1 = (int(round(b[0])), int(round(b[1])))
        cv2.arrowedLine(out, p0, p1, color, thickness, tipLength=0.35, line_type=cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/tingting/.tmp/official_viz_sample_ep0.npz")
    ap.add_argument("--hdf5", required=True, help="the LIBERO episode hdf5 (for real RGB frames)")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_flow_overlay")
    ap.add_argument("--topk", type=int, default=300, help="draw the K most-moving points")
    ap.add_argument("--upscale", type=int, default=3, help="render scale for crisp arrows")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.cache, allow_pickle=True)
    pred = z["__pred_scene_flows__"]          # [T,Ns,3] absolute world positions
    robot = z["robot_flows"] if "robot_flows" in z.files else None  # [T,Nr,3] (FK, world)
    K = z["cam0_intrinsic"].astype(np.float64)
    E = z["cam0_extrinsic"].astype(np.float64)
    T, Ns, _ = pred.shape

    # real agentview RGB at the SAME full-demo linspace timesteps used for the action
    with h5py.File(args.hdf5, "r") as f:
        rgb_all = np.asarray(f["data"][args.demo]["obs"]["agentview_rgb"][:])  # [F,128,128,3], opengl-flipped
    F = rgb_all.shape[0]
    ti = np.linspace(0, F - 1, T).round().astype(int)
    rgb_seq = rgb_all[ti][:, ::-1, :, :]      # un-flip (opengl) -> upright, [T,128,128,3]

    # the cache RGB is 180x320 (PW payload); project into THAT frame, then overlay on a resized real frame
    Hc, Wc = z["cam0_initial_rgb"].shape[:2]  # 180,320 (projection target)
    up = args.upscale

    # top-K moving points (by total horizon displacement) — the ones worth drawing
    disp_total = np.linalg.norm(pred[-1] - pred[0], axis=-1)
    kp = np.argsort(disp_total)[-args.topk:]

    frames = []
    for t in range(T):
        # background = real agentview at this timestep, resized to projection frame, upscaled
        bg = cv2.resize(rgb_seq[t], (Wc, Hc), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(bg, (Wc * up, Hc * up), interpolation=cv2.INTER_NEAREST)
        canvas = np.ascontiguousarray(bg[:, :, ::-1])  # RGB->BGR for cv2 draw

        uv0, z0 = project(pred[0][kp], K, E)
        uvt, zt = project(pred[t][kp], K, E)
        vis = (z0 > 1e-4) & (zt > 1e-4)
        # green = predicted cumulative flow (t0 -> t) for the moving points
        canvas = draw_flow(canvas, uv0 * up, uvt * up, vis, color=(0, 220, 0), thickness=1)
        # magenta = robot (FK) points at frame t — the actual arm motion (GT reference)
        if robot is not None:
            ti = min(t, robot.shape[0] - 1)
            ruv, rz = project(robot[ti], K, E)
            for (a, m) in zip(ruv * up, rz > 1e-4):
                if m:
                    cv2.circle(canvas, (int(a[0]), int(a[1])), 1, (255, 0, 255), -1)
        cv2.putText(canvas, f"frame {t+1}/{T}  green=pred flow  magenta=robot(GT arm)",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rgb_out = canvas[:, :, ::-1]  # back to RGB
        frames.append(rgb_out)
        cv2.imwrite(os.path.join(args.out_dir, f"frame_{t:02d}.png"), canvas)

    gif = os.path.join(args.out_dir, "flow_overlay.gif")
    imageio.mimsave(gif, frames, fps=args.fps, loop=0)
    print(f"[overlay] wrote {T} frames + {gif}", flush=True)
    print(f"[overlay] arrows = predicted point-flow (t0->t) projected on the REAL agentview video", flush=True)


if __name__ == "__main__":
    main()
