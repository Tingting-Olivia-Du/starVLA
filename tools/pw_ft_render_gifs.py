# [LangPointWorld] Render a GIF per FINETUNED-model prediction npz (from pw_ft_imagine_episodes.py).
# Projects the world-frame predicted scene cloud into cam0 at each frame, overlays predicted flow
# arrows (green) on the top-K movers, the robot cloud (magenta), and the GT object trajectory
# (yellow, projected). Lets you SEE the finetuned model carry the grasped object.
import argparse, glob, os
import numpy as np
import cv2
import imageio.v2 as imageio


def project(points_world, K, E_w2c):
    ones = np.ones((points_world.shape[0], 1))
    cam = (E_w2c @ np.concatenate([points_world, ones], -1).T).T[:, :3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def splat(canvas, uv, z, colors, radius=2, brighten=1.4):
    H, W = canvas.shape[:2]
    order = np.argsort(-z)  # far first
    for i in order:
        x, y = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= x < W and 0 <= y < H and z[i] > 1e-4:
            c = np.clip(colors[i].astype(np.float32) * brighten, 0, 255).astype(np.uint8)
            cv2.circle(canvas, (x, y), radius, (int(c[2]), int(c[1]), int(c[0])), -1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/workspace/tingting/.tmp/s0_ftviz/pred_*.npz")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/s0_ftviz/gifs")
    ap.add_argument("--topk", type=int, default=250)
    ap.add_argument("--upscale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for npz in sorted(glob.glob(args.glob)):
        tag = os.path.basename(npz)[5:-4]
        z = np.load(npz, allow_pickle=True)
        pred = z["__pred_scene_flows__"]                 # [T,Ns,3] world
        colors = z["_scene_colors_u8"]                   # [Ns,3] u8
        robot = z["robot_flows"] if "robot_flows" in z.files else None
        K = z["cam0_intrinsic"].astype(np.float64)
        E = z["cam0_extrinsic"].astype(np.float64)       # world->cam
        obj_traj = z["obj_traj_world"]                   # [T,3] GT object center (world)
        gt_disp = float(z["gt_disp"])
        T, Ns, _ = pred.shape
        Hc, Wc = 180, 320
        up = args.upscale; H, W = Hc * up, Wc * up
        Ku = K.copy(); Ku[:2, :] *= up

        disp_total = np.linalg.norm(pred[-1] - pred[0], axis=-1)
        kp = np.argsort(disp_total)[-args.topk:]
        pw_p90 = np.percentile(disp_total[kp], 90)

        frames = []
        for t in range(T):
            canvas = np.full((H, W, 3), 40, np.uint8)
            uv, zc = project(pred[t], Ku, E)
            canvas = splat(canvas, uv, zc, colors[:Ns], radius=2)
            if robot is not None:
                ti = min(t, robot.shape[0] - 1)
                ruv, rz = project(robot[ti], Ku, E)
                rcol = np.tile(np.array([[255, 0, 255]], np.uint8), (robot.shape[1], 1))
                canvas = splat(canvas, ruv, rz, rcol, radius=2, brighten=1.0)
            # predicted flow arrows (green)
            uv0, z0 = project(pred[0][kp], Ku, E)
            uvt, zt = project(pred[t][kp], Ku, E)
            vis = (z0 > 1e-4) & (zt > 1e-4)
            for a, b, m in zip(uv0, uvt, vis):
                if m:
                    cv2.arrowedLine(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                                    (0, 230, 0), 1, tipLength=0.3, line_type=cv2.LINE_AA)
            # GT object trajectory (yellow line + current dot)
            ouv, oz = project(obj_traj[:t + 1], Ku, E)
            for i in range(1, len(ouv)):
                cv2.line(canvas, (int(ouv[i - 1, 0]), int(ouv[i - 1, 1])),
                         (int(ouv[i, 0]), int(ouv[i, 1])), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(ouv[-1, 0]), int(ouv[-1, 1])), 4, (0, 255, 255), -1)
            cv2.putText(canvas, f"{tag}  f{t+1}/{T}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "green=pred flow  yellow=GT obj  magenta=robot", (6, H - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"GT disp {gt_disp:.2f}m  pred-mover p90 {pw_p90:.2f}m", (6, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
            frames.append(canvas[:, :, ::-1].copy())
        gif = os.path.join(args.out_dir, f"{tag}.gif")
        imageio.mimsave(gif, frames, fps=args.fps, loop=0)
        print(f"[gif] {tag}: GT {gt_disp:.2f}m pred-p90 {pw_p90:.2f}m -> {gif}", flush=True)


if __name__ == "__main__":
    main()
