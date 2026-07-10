# [LangPointWorld] Official viser (URDF-mesh robot) for the FINETUNED PW on a LIBERO sim-render
# episode. Runs inference in WORLD frame (the finetuned model's convention, S0-fixed), then
# transforms the imagined scene_flows + camera extrinsic into the ROBOT-BASE frame so they align
# with the base-origin URDF mesh the official PredictionVisualizer FK's from joint_positions.
# => full robot arm rendered as a mesh (not the sparse 512-pt FK cloud) + the finetuned prediction.
import argparse, glob, os, sys
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
_PW_ROOT = "/workspace/tingting/PointWorld"
sys.path.insert(0, _PW_ROOT)

from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim", required=True, help="sim-render npz (tools/pw_sim_render.py output)")
    ap.add_argument("--viewer-port", type=int, default=8080)
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--single", action="store_true", help="single agentview (default DUAL)")
    ap.add_argument("--foveated", action="store_true", help="object-foveated official preprocessing")
    args = ap.parse_args()

    base_name = os.path.basename(args.sim)[:-4]
    demo = base_name.split("_demo_")[1]; task = base_name.rsplit("_demo_", 1)[0]
    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"

    sr = np.load(args.sim, allow_pickle=True)
    names = list(sr["obj_names"]); op = sr["obj_poses"]
    base_T = op[0][names.index("robot0_base")]         # base -> world
    base_inv = np.linalg.inv(base_T)                   # world -> base

    with h5py.File(H, "r") as f:
        o = f["data"][f"demo_{demo}"]["obs"]
        joints = np.asarray(o["joint_states"][:], np.float64)
        grip = np.asarray(o["gripper_states"][:], np.float64)

    builder = LiberoDataDictBuilder(domain="droid", device="cuda", max_scene_points=30000)
    teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")
    # BASE frame + OFFICIAL PREPROCESSING (center+voxel1.5cm+12k+normalize) — this is the input the
    # finetuned PW performs BEST on (57% obj-motion vs 22% raw). Dual-view, robot points removed (bug
    # fix). The official center_shift adds __shift_amount__; the official visualizer uses it to place
    # the base-origin URDF mesh so the mesh aligns with the (now-centered) scene cloud.
    cams = ("agentview",) if args.single else ("agentview", "robot0_eye_in_hand")
    dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=cams, frame="base",
                                official_preprocess=True, official_importance=args.foveated)
    pred = teacher.imagine_from_datadict(dd)["scene_flows"].numpy()   # [T,Ns,3] (centered base frame)
    shift = np.asarray(dd.get("__shift_amount__", np.zeros(3)), np.float32)
    obj_ratio = None
    names_l = list(sr["obj_names"]); op = sr["obj_poses"]
    cand = [n for n in names_l if not n.startswith(("robot0", "gripper", "mount")) and n not in ("floor", "table", "world")]
    if cand:
        d = {n: np.linalg.norm(op[-1][names_l.index(n)][:3, 3] - op[0][names_l.index(n)][:3, 3]) for n in cand}
        obj = max(d, key=d.get)
        # object anchor in the same centered-base frame: world -> base -> +shift
        opos_b = (base_inv[:3, :3] @ op[0][names_l.index(obj)][:3, 3]) + base_inv[:3, 3] + shift
        near = np.linalg.norm(pred[0] - opos_b, axis=-1) < 0.08
        if near.sum():
            obj_ratio = float(np.percentile(np.linalg.norm(pred[-1][near] - pred[0][near], -1), 90)) / max(d[obj], 1e-3)

    ti = sr["ti"]
    sample_dict = {
        "__domain__": "droid",
        "scene_flows": dd["scene_flows"].astype(np.float32),
        "scene_colors": dd["_scene_colors_u8"][: dd["scene_flows"].shape[1]].astype(np.uint8),
        "scene_exists": dd["scene_exists"],
        "scene_supervised_mask": np.ones_like(dd["scene_exists"], dtype=bool),
        "robot_flows": dd["robot_flows"].astype(np.float32),
        "robot_exists": dd["robot_exists"],
        "joint_positions": joints[ti][:T_MODEL].astype(np.float64),   # URDF mesh FK (base origin)
        "gripper_positions": np.asarray(grip[ti][:T_MODEL], np.float64).mean(axis=1),
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "__shift_amount__": shift,     # official visualizer places the URDF mesh with this shift
    }
    # pass ALL camera payloads (cam0=agentview, cam1=wrist) so the official viz shows BOTH views
    for ci in range(len(cams)):
        if f"cam{ci}_initial_rgb" in dd:
            sample_dict[f"cam{ci}_initial_rgb"] = dd[f"cam{ci}_initial_rgb"]
            sample_dict[f"cam{ci}_initial_depth"] = dd[f"cam{ci}_initial_depth"]
            sample_dict[f"cam{ci}_intrinsic"] = dd[f"cam{ci}_intrinsic"]
            sample_dict[f"cam{ci}_extrinsic"] = dd[f"cam{ci}_extrinsic"].astype(np.float64)
    predictions = {"scene_flows": pred.astype(np.float32)}
    print(f"[ft-official-viz] pred obj-motion ratio (near-obj p90 / GT) = "
          f"{obj_ratio if obj_ratio is not None else float('nan'):.2f}", flush=True)

    os.chdir(_PW_ROOT)
    from visualization.prediction_viz.sample import build_sample_from_dictionary
    from visualization.prediction_viz.visualizer import PredictionVisualizer
    from visualization.prediction_viz.config import PredictionVisualizerConfig
    from utils import resolve_robot_urdf

    sample = build_sample_from_dictionary(sample_dict=sample_dict, predictions=predictions)
    cfg = PredictionVisualizerConfig(); cfg.viewer_port = args.viewer_port; cfg.viewer_host = "0.0.0.0"
    # Use LIBERO's real PANDA gripper URDF for the mesh render (not PW's Robotiq droid URDF) so the
    # rendered arm matches the actual robot; robot_flows already verified aligned to real gripper (8099).
    from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder as _B
    urdf = args.urdf or _B._PANDA_URDF
    viz = PredictionVisualizer(cfg, urdf_path=urdf)
    print(f"[ft-official-viz] robot mesh URDF = {urdf}", flush=True)
    print(f"[ft-official-viz] {task} demo_{demo} | launching http://localhost:{args.viewer_port}", flush=True)
    result = viz.visualize(sample, launch_viewer=True)
    import time
    print(f"[ft-official-viz] READY http://localhost:{args.viewer_port} (full URDF-mesh robot + finetuned pred). Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
