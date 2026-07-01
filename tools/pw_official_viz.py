# [LangPointWorld] Drive PointWorld's OFFICIAL viser visualizer (visualization/prediction_viz)
# with our LIBERO-imagined scene flow. Rebuilds the bridge data_dict + colors as the official
# sample_dict, feeds the teacher's imagined scene_flows as `predictions`, launches the viewer.
# Uses the vendored NVIDIA viz (Apache-2.0) unchanged — we only supply the sample.
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

_PW_ROOT = "/workspace/tingting/PointWorld"
sys.path.insert(0, _PW_ROOT)

from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder, unproject_depth_to_points
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

T_MODEL = 11


def build_official_sample_dict(builder, geo, teacher, hdf5):
    """Returns (sample_dict, predictions) for build_sample_from_dictionary."""
    demo, t0 = "demo_0", 0
    dp = geo.depth_provider(hdf5, demo)
    K, E = geo.intrinsics_extrinsics(hdf5, demo, t0)
    s = build_pw_sample(hdf5, demo, t0, depth_provider=dp, domain="droid", intrinsics=K, extrinsics=E)
    with h5py.File(hdf5, "r") as f:
        obs = f["data"][demo]["obs"]
        joints = np.asarray(obs["joint_states"][:], np.float64)
        grip = np.asarray(obs["gripper_states"][:], np.float64)
    dd = builder.build(s, joints, grip, horizon=T_MODEL)
    out = teacher.imagine_from_datadict(dd)

    # per-point colors for the scene cloud
    depth = s["agentview_initial_depth"]; rgb = s["agentview_initial_rgb"]
    Kb = s["agentview_intrinsic"].copy()
    Kb[0, :] *= (depth.shape[1] / 128); Kb[1, :] *= (depth.shape[0] / 128)
    _, colors, _ = unproject_depth_to_points(depth, Kb, s["agentview_extrinsic"], rgb=rgb,
                                             max_points=builder.max_scene_points)
    Ns = dd["scene_flows"].shape[1]
    colors = (np.clip(colors[:Ns], 0, 1) * 255).astype(np.uint8)

    sample_dict = {
        "__domain__": "droid",
        "scene_flows": dd["scene_flows"],          # GT-slot = static init (no GT scene track on LIBERO)
        "scene_colors": colors,
        "scene_exists": dd["scene_exists"],
        # all reconstructed points are "supervised" for viz purposes (no data-branch drop mask)
        "scene_supervised_mask": np.ones_like(dd["scene_exists"], dtype=bool),
        "robot_flows": dd["robot_flows"],
        "robot_exists": dd["robot_exists"],
        "joint_positions": joints[:T_MODEL],
        "gripper_positions": np.asarray(grip[:T_MODEL], np.float64).mean(axis=1),
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "cam0_initial_rgb": dd["cam0_initial_rgb"],
        "cam0_initial_depth": dd["cam0_initial_depth"],
        "cam0_intrinsic": dd["cam0_intrinsic"],
        "cam0_extrinsic": dd["cam0_extrinsic"],
    }
    predictions = {"scene_flows": out["scene_flows"].numpy()}   # the teacher's imagination
    return sample_dict, predictions, os.path.basename(hdf5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libero-glob", required=True)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--viewer-port", type=int, default=8080)
    ap.add_argument("--rebuild", action="store_true", help="force re-run models even if cache exists")
    args = ap.parse_args()

    os.chdir(_PW_ROOT)  # official viz + URDF paths are repo-root-relative
    from visualization.prediction_viz.sample import build_sample_from_dictionary
    from visualization.prediction_viz.visualizer import PredictionVisualizer
    from visualization.prediction_viz.config import PredictionVisualizerConfig
    from utils import resolve_robot_urdf

    files = sorted(glob.glob(args.libero_glob))
    hdf5 = files[args.episode_index]
    print(f"[official-viz] episode: {os.path.basename(hdf5)}", flush=True)

    # Cache the built sample_dict+predictions so viz iterations skip the ~3min model reload.
    cache = f"/workspace/tingting/.tmp/official_viz_sample_ep{args.episode_index}.npz"
    if os.path.exists(cache) and not args.rebuild:
        print(f"[official-viz] loading cached sample from {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        sample_dict = {k: z[k] for k in z.files if k != "__pred_scene_flows__"}
        sample_dict["__domain__"] = "droid"
        sample_dict["joint_names"] = [f"panda_joint{i}" for i in range(1, 8)]
        predictions = {"scene_flows": z["__pred_scene_flows__"]}
    else:
        geo = VGGTGeometryProvider(device="cuda")
        builder = LiberoDataDictBuilder(domain="droid", device="cuda")
        teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")
        sample_dict, predictions, name = build_official_sample_dict(builder, geo, teacher, hdf5)
        savez = {k: v for k, v in sample_dict.items() if isinstance(v, np.ndarray)}
        savez["__pred_scene_flows__"] = predictions["scene_flows"]
        np.savez(cache, **savez)
        print(f"[official-viz] cached sample -> {cache}", flush=True)

    sample = build_sample_from_dictionary(sample_dict=sample_dict, predictions=predictions)
    cfg = PredictionVisualizerConfig()
    cfg.viewer_port = args.viewer_port
    cfg.viewer_host = "0.0.0.0"
    urdf = os.path.join(_PW_ROOT, resolve_robot_urdf("droid"))
    viz = PredictionVisualizer(cfg, urdf_path=urdf)
    print(f"[official-viz] launching viewer on http://localhost:{args.viewer_port} ...", flush=True)
    viz.visualize(sample, launch_viewer=True)


if __name__ == "__main__":
    main()
