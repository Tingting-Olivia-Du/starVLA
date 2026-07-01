# [LangPointWorld] Generate ACCURATE per-object GT scene flow for PW finetuning, from a sim-render
# npz (real depth + real K/E + per-pixel body seg + per-object GT pose trajectory).
#
# Correctness (this is the whole point — GT must be exact):
#   1. Unproject t=0 depth -> 3D surface points in the WORLD frame (real K/E).
#   2. Assign each point to its object via the per-pixel body-id SEGMENTATION (exact, not nearest-
#      center). Points on floor/world/table are background (static); robot points are dropped
#      (robot is the action input, not scene). Task/graspable objects get RIGID-BODY GT flow.
#   3. Each object point is a RIGID point on that body: express it in the body's t=0 local frame
#      (P_local = inv(T_obj[0]) @ P_world0), then transform by the body's GT pose at each frame
#      (P_world[t] = T_obj[t] @ P_local). This is exact under the rigid-body assumption MuJoCo uses.
#   4. Static/background points keep their t=0 position for all frames (GT: they don't move).
# Output = scene_flows[T,Ns,3] (world frame) = the exact GT PW must learn to predict.
import argparse, json, os
import numpy as np


def unproject(depth, K, E_cam2world, flip=True):
    d = depth[::-1, :].copy() if flip else depth.copy()   # opengl flip
    H, W = d.shape
    ys, xs = np.nonzero(np.isfinite(d) & (d > 1e-4) & (d < 5))
    z = d[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], -1)
    Pw = (E_cam2world @ np.concatenate([pc, np.ones((len(pc), 1))], -1).T).T[:, :3]
    return Pw.astype(np.float64), (ys, xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=8192)
    args = ap.parse_args()

    z = np.load(args.sim, allow_pickle=True)
    names = list(z["obj_names"])
    b2n = json.loads(str(z["bodyid_to_name"]))
    op = z["obj_poses"]                                    # [T,25,4,4] world
    T = op.shape[0]
    name2idx = {n: i for i, n in enumerate(names)}

    depth0 = z[f"{args.cam}_depth"][0]
    K = z[f"{args.cam}_K"][0]
    E_c2w = z[f"{args.cam}_E_cam2world"][0]
    seg0 = z[f"{args.cam}_seg_bodyid"][0][::-1, :]         # flip to match unproject's flip

    Pw0, (ys, xs) = unproject(depth0, K, E_c2w, flip=True)
    pt_body = seg0[ys, xs]                                 # body id per surface point

    # subsample to max_points (keep object points preferentially)
    is_obj = np.array([b2n.get(str(int(b)), "") not in ("floor", "world", "table", "")
                       and not b2n.get(str(int(b)), "").startswith(("robot0", "gripper", "mount"))
                       for b in pt_body])
    is_robot = np.array([b2n.get(str(int(b)), "").startswith(("robot0", "gripper", "mount"))
                         for b in pt_body])
    keep = ~is_robot                                       # drop robot points (action, not scene)
    Pw0, pt_body, is_obj = Pw0[keep], pt_body[keep], is_obj[keep]
    if len(Pw0) > args.max_points:
        # keep all object points + fill with background
        obj_i = np.where(is_obj)[0]; bg_i = np.where(~is_obj)[0]
        rng = np.random.default_rng(0)
        n_bg = max(0, args.max_points - len(obj_i))
        sel = np.concatenate([obj_i, rng.choice(bg_i, min(n_bg, len(bg_i)), replace=False)])
        Pw0, pt_body, is_obj = Pw0[sel], pt_body[sel], is_obj[sel]

    Ns = len(Pw0)
    scene_flows = np.tile(Pw0[None], (T, 1, 1))            # default: static (bg keeps t=0)
    # rigid-body GT flow for object points
    name_by_bid = {int(k): v for k, v in b2n.items()}
    for bid in np.unique(pt_body):
        nm = name_by_bid.get(int(bid), "")
        if nm not in name2idx:                            # only bodies we have poses for
            continue
        if nm in ("floor", "world", "table") or nm.startswith(("robot0", "gripper", "mount")):
            continue
        m = pt_body == bid
        oi = name2idx[nm]
        T0 = op[0][oi]; T0_inv = np.linalg.inv(T0)
        P_local = (T0_inv[:3, :3] @ Pw0[m].T).T + T0_inv[:3, 3]   # points in body local frame
        for t in range(T):
            Tt = op[t][oi]
            scene_flows[t][m] = (Tt[:3, :3] @ P_local.T).T + Tt[:3, 3]

    # sanity: object-point GT displacement should match the object's pose displacement
    report = {}
    for bid in np.unique(pt_body):
        nm = name_by_bid.get(int(bid), "")
        if nm not in name2idx or nm in ("floor", "world", "table") or nm.startswith(("robot0", "gripper", "mount")):
            continue
        m = pt_body == bid
        oi = name2idx[nm]
        pose_disp = np.linalg.norm(op[-1][oi][:3, 3] - op[0][oi][:3, 3])
        pt_disp = np.linalg.norm(scene_flows[-1][m] - scene_flows[0][m], axis=-1).mean()
        report[nm] = (int(m.sum()), round(float(pose_disp), 4), round(float(pt_disp), 4))

    colors = np.zeros((Ns, 3), np.uint8)  # optional; downstream can recolor
    np.savez(args.out, scene_flows=scene_flows.astype(np.float32),
             pt_body=pt_body.astype(np.int32), is_obj=is_obj,
             cam_K=K, cam_E_cam2world=E_c2w)
    print(f"[gt-flow] Ns={Ns} ({is_obj.sum()} object pts) -> {args.out}", flush=True)
    print("[gt-flow] SANITY (object pts): name | npts | pose_disp | GT_point_disp (should MATCH)", flush=True)
    for nm, (n, pd, ptd) in report.items():
        ok = "OK" if abs(pd - ptd) < 0.02 else "MISMATCH!"
        print(f"   {nm:24s} {n:5d} {pd:7.4f} {ptd:7.4f}  {ok}", flush=True)


if __name__ == "__main__":
    main()
