# [LangPointWorld] Dump full imagined scene-flow (+ colors, robot points, GT EE traj) per LIBERO
# episode for the viser 3D viewer. Reuses the verified imagine pipeline.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse
import glob
import os
import sys

import numpy as np
import torch
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")

from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder, unproject_depth_to_points
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

T_MODEL = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libero-glob", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_scene_flow")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(args.libero_glob))[: args.episodes]
    print(f"[dump] {len(files)} episodes; loading VGGT + teacher ...", flush=True)
    geo = VGGTGeometryProvider(device="cuda")
    builder = LiberoDataDictBuilder(domain="droid", device="cuda")
    teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")

    for fi, hdf5 in enumerate(files):
        demo, t0 = "demo_0", 0
        dp = geo.depth_provider(hdf5, demo)
        K, E = geo.intrinsics_extrinsics(hdf5, demo, t0)
        s = build_pw_sample(hdf5, demo, t0, depth_provider=dp, domain="droid", intrinsics=K, extrinsics=E)
        with h5py.File(hdf5, "r") as f:
            obs = f["data"][demo]["obs"]
            joints = np.asarray(obs["joint_states"][:], np.float64)
            grip = np.asarray(obs["gripper_states"][:], np.float64)
            ee_traj = np.asarray(obs["ee_pos"][:], np.float64)  # full GT EE trajectory (world)
        dd = builder.build(s, joints, grip, horizon=T_MODEL)
        out = teacher.imagine_from_datadict(dd)

        # recover per-point RGB colors for the scene cloud (re-unproject with rgb)
        depth = s["agentview_initial_depth"]; rgb = s["agentview_initial_rgb"]
        Kb = s["agentview_intrinsic"].copy()
        Kb[0, :] *= (depth.shape[1] / 128); Kb[1, :] *= (depth.shape[0] / 128)
        _, colors, _ = unproject_depth_to_points(depth, Kb, s["agentview_extrinsic"], rgb=rgb,
                                                  max_points=builder.max_scene_points)
        np.savez(os.path.join(args.out_dir, f"ep{fi}.npz"),
                 scene_flows=out["scene_flows"].numpy(),      # [T,Ns,3] absolute positions
                 scene_colors=colors[: out["scene_coord0"].shape[0]],  # [Ns,3] 0..1
                 robot_flows=dd["robot_flows"],               # [T,Nr,3]
                 ee_traj=ee_traj[:T_MODEL],                   # [T,3] GT EE
                 name=os.path.basename(hdf5))
        print(f"[dump] ep{fi} -> {os.path.basename(hdf5)}", flush=True)
    print("[dump] DONE", flush=True)


if __name__ == "__main__":
    main()
