# [LangPointWorld] Render the imagined scene point-flow as a self-consistent 2D GIF: splat the
# VGGT-reconstructed colored point cloud into the VGGT camera at each frame, overlay predicted
# flow arrows. Everything (points, colors, flow, camera) lives in ONE coordinate system, so there
# is no cross-camera misalignment (unlike projecting onto the raw LIBERO agentview, whose real
# camera differs from VGGT's estimated pose). Reads the cached imagine sample (no model reload).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse
import os

import numpy as np
import cv2
import imageio.v2 as imageio


def project(points_world, K, E_world_to_cam):
    ones = np.ones((points_world.shape[0], 1), points_world.dtype)
    cam = (E_world_to_cam @ np.concatenate([points_world, ones], -1).T).T[:, :3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def splat(canvas, uv, z, colors, radius=2, brighten=1.4):
    """Painter's-order point splat (far first) into an HxWx3 BGR canvas."""
    H, W = canvas.shape[:2]
    order = np.argsort(-z)  # far -> near
    for i in order:
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= x < W and 0 <= y < H and z[i] > 1e-4:
            c = np.clip(colors[i].astype(np.float32) * brighten, 0, 255).astype(int)
            cv2.circle(canvas, (x, y), radius, (int(c[2]), int(c[1]), int(c[0])), -1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/tingting/.tmp/official_viz_sample_ep0.npz")
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_flow_render")
    ap.add_argument("--topk", type=int, default=250)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--show-robot", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.cache, allow_pickle=True)
    pred = z["__pred_scene_flows__"]                 # [T,Ns,3] world
    colors = z["scene_colors"]                       # [Ns,3] uint8
    robot = z["robot_flows"] if "robot_flows" in z.files else None  # [T,Nr,3]
    K = z["cam0_intrinsic"].astype(np.float64)
    E = z["cam0_extrinsic"].astype(np.float64)
    Hc, Wc = z["cam0_initial_rgb"].shape[:2]
    T, Ns, _ = pred.shape
    up = args.upscale
    H, W = Hc * up, Wc * up
    Ku = K.copy(); Ku[:2, :] *= up                   # scale intrinsics for the upscaled canvas

    disp_total = np.linalg.norm(pred[-1] - pred[0], axis=-1)
    kp = np.argsort(disp_total)[-args.topk:]

    frames = []
    for t in range(T):
        canvas = np.full((H, W, 3), 40, np.uint8)    # gray bg (brighter than near-black)
        # 1) splat the reconstructed scene cloud at frame t (self-consistent camera)
        uv, zc = project(pred[t], Ku, E)
        canvas = splat(canvas, uv, zc, colors, radius=2)
        # 2) robot points (magenta)
        if args.show_robot and robot is not None:
            ti = min(t, robot.shape[0] - 1)
            ruv, rz = project(robot[ti], Ku, E)
            rcol = np.tile(np.array([[255, 0, 255]], np.uint8), (robot.shape[1], 1))
            canvas = splat(canvas, ruv, rz, rcol, radius=2, brighten=1.0)
        # 3) predicted flow arrows (t0 -> t) on the top-K movers (green)
        uv0, z0 = project(pred[0][kp], Ku, E)
        uvt, zt = project(pred[t][kp], Ku, E)
        vis = (z0 > 1e-4) & (zt > 1e-4)
        for a, b, m in zip(uv0, uvt, vis):
            if not m:
                continue
            cv2.arrowedLine(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                            (0, 230, 0), 1, tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.putText(canvas, f"frame {t+1}/{T}  green=pred flow  magenta=robot",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        frames.append(canvas[:, :, ::-1].copy())     # BGR->RGB
        cv2.imwrite(os.path.join(args.out_dir, f"frame_{t:02d}.png"), canvas)

    gif = os.path.join(args.out_dir, "flow_render.gif")
    imageio.mimsave(gif, frames, fps=args.fps, loop=0)
    print(f"[render] wrote {T} frames + {gif}", flush=True)


if __name__ == "__main__":
    main()
