# [LangPointWorld] SANITY-CHECK the GT scene flow that was PACKED into the droid HDF5 (not a
# recomputation) — read scene_flows straight from the finetune HDF5, project onto the real
# agentview video, color points by their per-frame displacement (moving=red-hot, static=blue).
# If the GT is correct: moving points sit exactly on the manipulated object and follow it; the rest
# stay put. Confirms the packed WDS supervision is accurate end-to-end.
import argparse, os
import numpy as np
import cv2
import h5py
import imageio.v2 as imageio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="packed droid-format HDF5")
    ap.add_argument("--sim", required=True, help="sim-render npz for the real RGB frames")
    ap.add_argument("--clip", default="0:11")
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_gt_sanity")
    ap.add_argument("--up", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        g = f[args.clip]["camera_0"]
        sf = np.asarray(g["scene_flows"])              # [T,Ns,3] world (the PACKED GT)
        K = np.asarray(g["intrinsic"])                 # (3,3) at 180x320
        E_w2c = np.asarray(g["extrinsic"])             # world->cam
    z = np.load(args.sim, allow_pickle=True)
    rgb = z[f"{args.cam}_rgb"][:, ::-1, :, :]          # un-flip real frames
    T, Ns, _ = sf.shape
    H0, W0 = rgb.shape[1:3]
    # K in HDF5 is at 180x320; project in that frame then map to the real-frame pixel grid
    Hc, Wc = 180, 320

    disp_total = np.linalg.norm(sf[-1] - sf[0], axis=-1)   # per-point total motion
    dmax = max(disp_total.max(), 1e-3)

    def project(P):
        cam = (E_w2c @ np.concatenate([P, np.ones((len(P), 1))], -1).T).T[:, :3]
        z_ = cam[:, 2]
        uv = (K @ cam.T).T
        uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
        # HDF5 K is at (180,320); scale to real (H0,W0)
        uv[:, 0] *= (W0 / Wc); uv[:, 1] *= (H0 / Hc)
        return uv, z_

    up = args.up
    frames = []
    for t in range(T):
        bg = cv2.resize(rgb[t], (W0 * up, H0 * up), interpolation=cv2.INTER_NEAREST)
        canvas = np.ascontiguousarray(bg[:, :, ::-1])
        uv, zc = project(sf[t])
        d_now = np.linalg.norm(sf[t] - sf[0], axis=-1)     # displacement so far
        for i in range(Ns):
            if zc[i] <= 1e-4:
                continue
            x, y = int(uv[i, 0] * up), int(uv[i, 1] * up)
            if not (0 <= x < W0 * up and 0 <= y < H0 * up):
                continue
            r = d_now[i] / dmax                            # 0..1 heat
            color = (int(255 * (1 - r)), 0, int(255 * r))  # BGR: static=blue, moving=red
            cv2.circle(canvas, (x, y), 1, color, -1)
        cv2.putText(canvas, f"frame {t+1}/{T}  RED=GT moving points  BLUE=static  (packed HDF5 scene_flows)",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        frames.append(canvas[:, :, ::-1])
        cv2.imwrite(os.path.join(args.out_dir, f"frame_{t:02d}.png"), canvas)
    imageio.mimsave(os.path.join(args.out_dir, "gt_sanity.gif"), frames, fps=2, loop=0)
    n_move = int((disp_total > 0.05).sum())
    print(f"[gt-sanity] {T} frames, {Ns} pts, {n_move} moving>5cm, max disp {dmax:.3f} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
