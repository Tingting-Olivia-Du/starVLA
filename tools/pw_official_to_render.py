# [LangPointWorld] Convert an official PointWorld episode (per-object mesh points x GT pose
# trajectory) into the render-npz format used by pw_flow_render_gif.py. This is GROUND-TRUTH flow
# (how the dataset encodes scene motion), for visual comparison against our LIBERO PW predictions.
# Part of the PointWorld point-flow distillation arm.
import argparse
import numpy as np
import h5py


def quat2R(q):  # q = (x,y,z,w)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--sample", default=None, help="frame-pair group key, e.g. 190:201")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    g = args.sample or list(f.keys())[0]
    ch = f[g]["camera_head"]
    lp = ch["local_scene_points"]
    lc = ch["local_scene_colors"]
    traj = ch["scene_mesh_trajectories"]
    objs = list(lp.keys())

    T = np.asarray(traj[objs[0]]).shape[0]
    all_flows, all_colors = [], []
    for o in objs:
        pts = np.asarray(lp[o]).astype(np.float32)          # [N,3] local mesh points
        cols = np.asarray(lc[o]).astype(np.uint8)           # [N,3]
        tr = np.asarray(traj[o]).astype(np.float64)         # [T,7] pos(3)+quat(4)
        # world position of each mesh point at each timestep: R(t) @ p + t(t)
        seq = np.zeros((T, pts.shape[0], 3), np.float32)
        for t in range(T):
            R = quat2R(tr[t, 3:7]); tvec = tr[t, :3]
            seq[t] = (R @ pts.T).T + tvec
        all_flows.append(seq)
        all_colors.append(cols)
    scene_flows = np.concatenate(all_flows, axis=1)          # [T, Ns, 3]
    scene_colors = np.concatenate(all_colors, axis=0)        # [Ns, 3]

    # robot points (if present) — official has world_to_robot; robot flow via urdf is not stored
    # per-point here, so we render scene GT flow only (the point of the comparison).
    save = {
        "__pred_scene_flows__": scene_flows,   # GT here, but same key so the render tool reads it
        "scene_colors": scene_colors,
        "cam0_intrinsic": np.asarray(ch["intrinsic"]).astype(np.float64),
        # extrinsic is world->cam? official stores 4x4 extrinsic; use as-is for projection
        "cam0_extrinsic": np.asarray(ch["extrinsic"]).astype(np.float64),
        "cam0_initial_depth": np.asarray(ch["initial_depth"]).astype(np.float32),
    }
    np.savez(args.out, **save)
    n_obj_pts = {o.replace("World__scene_0__", "")[:20]: np.asarray(lp[o]).shape[0] for o in objs}
    disp = np.linalg.norm(scene_flows[-1] - scene_flows[0], axis=-1)
    print(f"[official->render] sample {g}, {len(objs)} objects {n_obj_pts}", flush=True)
    print(f"[official->render] scene_flows {scene_flows.shape}, moved>1cm {(disp>0.01).mean():.1%}, "
          f"max {disp.max():.3f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
