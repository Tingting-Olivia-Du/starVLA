# [LangPointWorld] Diagnose whether PW predicts the GRASPED-object motion on LIBERO.
# Overlays, on the real agentview frames: GT grasped-object trajectory (red) vs PW's predicted
# motion of the scene points near that object (green). Makes the "PW misses the grasped object"
# finding visual. Reads the aligned dual-view sim sample + the sim-render object poses.
import argparse, os
import numpy as np
import cv2
import imageio.v2 as imageio


def project(P, K, E_w2c):
    ones = np.ones((P.shape[0], 1), P.dtype)
    cam = (E_w2c @ np.concatenate([P, ones], -1).T).T[:, :3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="/workspace/tingting/.tmp/sim_render_ep0.npz")
    ap.add_argument("--pred", default="/workspace/tingting/.tmp/sim_dualview_base_ep0.npz")
    ap.add_argument("--obj", default="alphabet_soup_1_main")
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_grasp_diag")
    ap.add_argument("--up", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sr = np.load(args.sim, allow_pickle=True)
    pr = np.load(args.pred, allow_pickle=True)
    names = list(sr["obj_names"])
    op = sr["obj_poses"]                         # [T,25,4,4] world
    base_T = op[0][names.index("robot0_base")]
    base_inv = np.linalg.inv(base_T)
    oidx = names.index(args.obj)

    # PW predicted scene flow (BASE frame). obj GT positions -> base frame.
    sf = pr["__pred_scene_flows__"]              # [T,Ns,3] base
    obj_traj_w = op[:, oidx, :3, 3]              # [T,3] world GT
    obj_traj_b = (base_inv[:3, :3] @ obj_traj_w.T).T + base_inv[:3, 3]

    # scene points near the object at t=0 (in base frame) -> their PW-predicted motion
    d0 = np.linalg.norm(sf[0] - obj_traj_b[0], axis=-1)
    near = d0 < 0.08
    print(f"[diag] object {args.obj}: {near.sum()} scene pts within 8cm at t=0", flush=True)

    # camera: agentview real (world->cam), points are base-frame -> base->cam = E_w2c @ base_T
    K = sr["agentview_K"][0]
    E_w2c = np.linalg.inv(sr["agentview_E_cam2world"][0])
    E_b2c = E_w2c @ base_T
    up = args.up
    Ku = K.copy(); Ku[:2, :] *= up
    T = sf.shape[0]

    rgb_seq = sr["agentview_rgb"][:, ::-1, :, :]   # un-flip
    H, W = rgb_seq.shape[1:3]

    frames = []
    for t in range(T):
        bg = cv2.resize(rgb_seq[t], (W*up, H*up), interpolation=cv2.INTER_NEAREST)
        canvas = np.ascontiguousarray(bg[:, :, ::-1])
        # GT object trajectory so far (red), base frame -> project
        uvg, zg = project(obj_traj_b[:t+1], Ku, E_b2c)
        for i in range(len(uvg)-1):
            cv2.line(canvas, tuple(uvg[i].astype(int)), tuple(uvg[i+1].astype(int)), (0,0,255), 2, cv2.LINE_AA)
        if len(uvg): cv2.circle(canvas, tuple(uvg[-1].astype(int)), 5, (0,0,255), -1)  # GT now
        # PW predicted motion of near-object points (green arrows t0->t)
        if near.sum() and t > 0:
            p0 = sf[0][near]; pt = sf[t][near]
            uv0, z0 = project(p0, Ku, E_b2c); uvt, zt = project(pt, Ku, E_b2c)
            for a, b in zip(uv0, uvt):
                cv2.arrowedLine(canvas, tuple(a.astype(int)), tuple(b.astype(int)), (0,220,0), 1, tipLength=0.3)
        cv2.putText(canvas, f"frame {t+1}/{T}  RED=GT object traj  GREEN=PW pred near object",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        gt_disp = np.linalg.norm(obj_traj_b[t]-obj_traj_b[0])
        pw_disp = np.linalg.norm(sf[t][near]-sf[0][near], axis=-1).mean() if near.sum() else 0
        cv2.putText(canvas, f"GT obj moved {gt_disp:.3f}m | PW pred {pw_disp:.3f}m",
                    (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        frames.append(canvas[:, :, ::-1])
        cv2.imwrite(os.path.join(args.out_dir, f"frame_{t:02d}.png"), canvas)
    imageio.mimsave(os.path.join(args.out_dir, "grasp_diag.gif"), frames, fps=2, loop=0)
    print(f"[diag] wrote {T} frames + grasp_diag.gif -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
