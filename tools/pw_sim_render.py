# [LangPointWorld] Render REAL metric depth + REAL camera K/E + object poses from a LIBERO demo
# by replaying HDF5 sim states in robosuite. Fixes the VGGT-vs-FK frame misalignment at the root:
# sim depth is metric and in the SAME world frame as the robot, so scene+robot align with zero
# error. Also yields per-object 6D poses -> the rigorous GT for the PW-PROBE gate.
# RUN IN THE `libero` CONDA ENV (py3.8, robosuite 1.4.1), NOT starVLA:
#   MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/tingting/envs/libero_extra_site \
#   /workspace/ghsun/miniconda3/envs/libero/bin/python tools/pw_sim_render.py --hdf5 ... --out ...
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse
import json
import os

import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--horizon", type=int, default=11, help="T action steps (linspace over demo)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", default="agentview,robot0_eye_in_hand")
    ap.add_argument("--img", type=int, default=128)
    args = ap.parse_args()

    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_camera_segmentation)

    cams = args.cameras.split(",")
    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        demo = data[args.demo]
        states = np.asarray(demo["states"])            # [F,110] full MuJoCo state
        F = states.shape[0]
        bddl_rel = data.attrs["bddl_file_name"] if "bddl_file_name" in data.attrs else \
            env_args["env_kwargs"]["bddl_file_name"]
    # resolve the bddl to a real absolute path (HDF5 stores a stale relative/prefixed path)
    bname = os.path.basename(bddl_rel)
    suite = os.path.basename(os.path.dirname(bddl_rel))
    bddl = f"/workspace/tingting/LIBERO/libero/libero/bddl_files/{suite}/{bname}"
    assert os.path.exists(bddl), f"bddl not found: {bddl}"

    # OffScreenRenderEnv sets controller/robot internally; only pass bddl + camera config.
    # camera_segmentations="instance" -> per-pixel object id for EXACT point-to-object assignment.
    kw = dict(bddl_file_name=bddl, camera_names=cams, camera_depths=True,
              camera_heights=args.img, camera_widths=args.img,
              camera_segmentations="instance")
    env = OffScreenRenderEnv(**kw)
    env.reset()
    sim = env.env.sim

    ti = np.linspace(0, F - 1, args.horizon).round().astype(int)
    frames = {c: {"rgb": [], "depth": [], "seg": []} for c in cams}
    obj_names = [n for n in sim.model.body_names if n not in ("world", "table")]
    obj_poses = []  # per-timestep {body: 4x4}
    # instance-seg id -> body-name map (robosuite instance seg ids index a per-env instance list;
    # we map via geom_bodyid so each seg id resolves to the MuJoCo body it belongs to).
    seg_id_to_body = {}

    # camera K/E are static (agentview) or per-frame (wrist) — capture per timestep to be safe
    Ks = {c: [] for c in cams}
    Es = {c: [] for c in cams}

    def get_real_depth(cam):
        # robosuite returns normalized depth in [0,1]; convert to metric using near/far
        obs = env.env._get_observations(force_update=True)
        d = obs[f"{cam}_depth"][..., 0]                # [H,W] normalized
        ext = sim.model.stat.extent
        near = sim.model.vis.map.znear * ext
        far = sim.model.vis.map.zfar * ext
        metric = near / (1 - d * (1 - near / far))     # robosuite depth linearization
        rgb = obs[f"{cam}_image"]                       # [H,W,3] (flipped later by adapter)
        return rgb, metric.astype(np.float32)

    for fi, t in enumerate(ti):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        for c in cams:
            rgb, dep = get_real_depth(c)
            frames[c]["rgb"].append(rgb)
            frames[c]["depth"].append(dep)
            Ks[c].append(get_camera_intrinsic_matrix(sim, c, args.img, args.img))
            Es[c].append(get_camera_extrinsic_matrix(sim, c))  # camera->world (4x4)
            # per-pixel body-id map for EXACT point-to-object assignment (all frames; we use t=0
            # for binding but store all so either can be used). seg[...,1]=geom_id -> geom_bodyid.
            geom_seg = get_camera_segmentation(sim, c, args.img, args.img)[..., 1]
            body_seg = np.full(geom_seg.shape, -1, np.int32)
            for g in np.unique(geom_seg):
                if g < 0:
                    continue
                body_seg[geom_seg == g] = int(sim.model.geom_bodyid[g])
            frames[c]["seg"].append(body_seg)
        poses = {}
        for b in obj_names:
            bid = sim.model.body_name2id(b)
            T = np.eye(4)
            T[:3, 3] = sim.data.body_xpos[bid]
            T[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
            poses[b] = T
        obj_poses.append(poses)

    # body-id -> name map (for resolving seg ids to object names downstream)
    all_bids = set()
    for c in cams:
        for s in frames[c]["seg"]:
            all_bids.update(int(b) for b in np.unique(s) if b >= 0)
    bodyid_names = {int(b): sim.model.body_id2name(int(b)) for b in all_bids}

    save = {"ti": ti, "obj_names": np.array(obj_names, dtype=object),
            "bodyid_to_name": np.array(json.dumps(bodyid_names), dtype=object)}
    for c in cams:
        save[f"{c}_rgb"] = np.stack(frames[c]["rgb"], 0)          # [T,H,W,3]
        save[f"{c}_depth"] = np.stack(frames[c]["depth"], 0)      # [T,H,W] metric
        save[f"{c}_K"] = np.stack(Ks[c], 0)                       # [T,3,3]
        save[f"{c}_E_cam2world"] = np.stack(Es[c], 0)             # [T,4,4]
        save[f"{c}_seg_bodyid"] = np.stack(frames[c]["seg"], 0)   # [T,H,W] body id per pixel
    # object poses as [T, n_obj, 4,4]
    save["obj_poses"] = np.stack([np.stack([obj_poses[i][b] for b in obj_names], 0)
                                  for i in range(len(ti))], 0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **save)
    print(f"[sim-render] {len(ti)} frames, cams={cams}, {len(obj_names)} objects -> {args.out}", flush=True)
    print(f"[sim-render] agentview depth range "
          f"{save['agentview_depth'].min():.3f}~{save['agentview_depth'].max():.3f} (metric)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
