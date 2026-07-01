# [LangPointWorld] Pack a LIBERO sim-render npz into the OFFICIAL PointWorld DROID-format HDF5.
# DROID domain (unlike BEHAVIOR) uses DIRECT world-frame `scene_flows` (verified via
# data_integrity_check: droid requires camera_N/{scene_flows, scene_colors, scene_normals,
# scene_visibility, scene_depth_valid_mask, initial_rgb/depth/intrinsic/extrinsic} + top-level
# {gripper_open, gripper_pose, joint_positions, gripper_positions}). This matches our accurate,
# 3-way-verified GT scene flow directly — no per-object mesh intermediate needed.
#
# scene_flows are built EXACTLY like the verified pw_make_gt_flow: unproject t=0 depth -> assign to
# objects by per-pixel body seg -> rigid-body transform object points by GT pose trajectory; static
# background keeps t=0; robot points dropped.
import argparse, json, os
import numpy as np
import h5py
import cv2

_PW_HW = (180, 320)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--libero-hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--clip-key", default="0:11")
    ap.add_argument("--max-points", type=int, default=8192)
    args = ap.parse_args()

    z = np.load(args.sim, allow_pickle=True)
    names = list(z["obj_names"]); name2idx = {n: i for i, n in enumerate(names)}
    b2n = json.loads(str(z["bodyid_to_name"])); name_by_bid = {int(k): v for k, v in b2n.items()}
    op = z["obj_poses"]; T = op.shape[0]
    depth = z[f"{args.cam}_depth"]; rgb = z[f"{args.cam}_rgb"]; K = z[f"{args.cam}_K"]
    E_c2w = z[f"{args.cam}_E_cam2world"]; seg = z[f"{args.cam}_seg_bodyid"]
    H0, W0 = depth.shape[1:]

    # t=0 unproject -> world points + seg assignment.
    # CRITICAL flip contract (verified: object centroid aligns to sim object center only this way):
    # depth+rgb are opengl-flipped (need [::-1] for correct y-up unprojection), but the segmentation
    # from get_camera_segmentation is aligned to the UN-flipped image — so seg must NOT be flipped.
    # both-flip gave centroid offset 0.173, depth-flip-only gave 0.038 (correct).
    d0 = depth[0][::-1, :]; rgb0 = rgb[0][::-1, :, :]; seg0 = seg[0]           # seg NOT flipped
    ys, xs = np.nonzero(np.isfinite(d0) & (d0 > 1e-4) & (d0 < 5))
    zz = d0[ys, xs]; K0 = K[0]; fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    pc = np.stack([(xs - cx) / fx * zz, (ys - cy) / fy * zz, zz], -1)
    Pw0 = (E_c2w[0] @ np.concatenate([pc, np.ones((len(pc), 1))], -1).T).T[:, :3]
    body = seg0[ys, xs]; cols = rgb0[ys, xs]

    def role(nm):
        if nm.startswith(("robot0", "gripper", "mount")):
            return "robot"
        if nm in name2idx and nm not in ("floor", "world", "table"):
            return "obj"
        return "bg"
    roles = np.array([role(name_by_bid.get(int(b), "")) for b in body])
    keep = roles != "robot"                                # drop robot (action, not scene)
    Pw0, body, cols, roles = Pw0[keep], body[keep], cols[keep], roles[keep]
    # subsample keeping all object points
    if len(Pw0) > args.max_points:
        oi = np.where(roles == "obj")[0]; bi = np.where(roles != "obj")[0]
        rng = np.random.default_rng(0)
        sel = np.concatenate([oi, rng.choice(bi, max(0, args.max_points - len(oi)), replace=False)])
        Pw0, body, cols, roles = Pw0[sel], body[sel], cols[sel], roles[sel]
    Ns = len(Pw0)

    # rigid-body GT scene flow (world frame)
    scene_flows = np.tile(Pw0[None], (T, 1, 1)).astype(np.float32)
    for bid in np.unique(body):
        nm = name_by_bid.get(int(bid), "")
        if role(nm) != "obj":
            continue
        m = body == bid; oi = name2idx[nm]
        T0inv = np.linalg.inv(op[0][oi])
        Plocal = (T0inv[:3, :3] @ Pw0[m].T).T + T0inv[:3, 3]
        for t in range(T):
            Tt = op[t][oi]
            scene_flows[t][m] = (Tt[:3, :3] @ Plocal.T).T + Tt[:3, 3]

    # all scene modalities must share (T, Ns) leading dims (integrity contract)
    scene_colors = np.tile(cols.astype(np.uint8)[None], (T, 1, 1))          # (T,Ns,3)
    scene_normals = np.zeros((T, Ns, 3), np.int8)                           # (T,Ns,3) quantized zeros
    scene_visibility = np.ones((T, Ns), bool)                              # (T,Ns)
    scene_depth_valid_mask = np.ones((T, Ns), bool)                        # (T,Ns)

    # camera payloads resized to (180,320)
    W1, H1 = _PW_HW[1], _PW_HW[0]
    Kp = K0.copy(); Kp[0, :] *= (W1 / W0); Kp[1, :] *= (H1 / H0)
    rgb_p = cv2.resize(rgb0, (W1, H1), interpolation=cv2.INTER_AREA)
    dep_p = cv2.resize(d0, (W1, H1), interpolation=cv2.INTER_NEAREST)
    dep_u16 = np.clip(dep_p * 1000.0, 0, 65535).astype(np.uint16)

    # robot fields from LIBERO hdf5 at the sim linspace timesteps
    ti = z["ti"]
    with h5py.File(args.libero_hdf5, "r") as lf:
        o = lf["data"][args.demo]["obs"]
        joints = np.asarray(o["joint_states"])[ti]; grip = np.asarray(o["gripper_states"])[ti]
        ee_pos = np.asarray(o["ee_pos"])[ti]; ee_ori = np.asarray(o["ee_ori"])[ti]
    gpos = grip.mean(axis=1).astype(np.float32)

    def eul2q(e):
        cx_, cy_, cz_ = np.cos(np.asarray(e) * 0.5); sx_, sy_, sz_ = np.sin(np.asarray(e) * 0.5)
        return np.array([sx_*cy_*cz_ - cx_*sy_*sz_, cx_*sy_*cz_ + sx_*cy_*sz_,
                         cx_*cy_*sz_ - sx_*sy_*cz_, cx_*cy_*cz_ + sx_*sy_*sz_])
    gp = np.stack([np.concatenate([ee_pos[t], eul2q(ee_ori[t])]) for t in range(T)], 0).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        g = f.create_group(args.clip_key)
        cam = g.create_group("camera_0")
        cam.create_dataset("scene_flows", data=scene_flows)                 # (T,Ns,3) world
        cam.create_dataset("scene_colors", data=scene_colors)              # (Ns,3)
        cam.create_dataset("scene_normals", data=scene_normals)            # (Ns,3) int8
        cam.create_dataset("scene_visibility", data=scene_visibility)      # (T,Ns)
        cam.create_dataset("scene_depth_valid_mask", data=scene_depth_valid_mask)  # (T,Ns)
        ok, buf = cv2.imencode(".jpg", rgb_p[:, :, ::-1])
        ds = cam.create_dataset("initial_rgb", (1,), dtype=h5py.special_dtype(vlen=np.uint8)); ds[0] = np.frombuffer(buf.tobytes(), np.uint8)
        cam.create_dataset("initial_depth", data=dep_u16)
        cam.create_dataset("intrinsic", data=Kp.astype(np.float32))
        cam.create_dataset("extrinsic", data=np.linalg.inv(E_c2w[0]).astype(np.float32))  # world->cam
        cam.create_dataset("extrinsic_trajectory", data=np.stack([np.linalg.inv(E_c2w[min(t,T-1)]) for t in range(T)],0).astype(np.float32))
        # top-level robot fields (droid contract)
        g.create_dataset("joint_positions", data=joints.astype(np.float32))
        g.create_dataset("gripper_positions", data=gpos)
        g.create_dataset("gripper_pose", data=gp)
        g.create_dataset("gripper_open", data=(gpos[:, None] > gpos.mean()))
    obj_ct = int((roles == "obj").sum())
    print(f"[pack-droid] {args.out} clip={args.clip_key} Ns={Ns} ({obj_ct} obj pts) scene_flows{scene_flows.shape}", flush=True)
    # self-check: object-point disp vs pose disp
    for bid in np.unique(body):
        nm = name_by_bid.get(int(bid), "")
        if role(nm) != "obj":
            continue
        m = body == bid; oi = name2idx[nm]
        pd = np.linalg.norm(op[-1][oi][:3,3]-op[0][oi][:3,3]); ptd = np.linalg.norm(scene_flows[-1][m]-scene_flows[0][m],axis=-1).mean()
        if pd > 0.05:
            print(f"   {nm}: pose_disp {pd:.3f} GT_pt_disp {ptd:.3f} {'OK' if abs(pd-ptd)<0.03 else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    main()
