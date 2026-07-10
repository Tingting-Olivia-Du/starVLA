# [LangPointWorld] Run the FINETUNED PW on several LIBERO episodes and dump a prediction npz per
# episode (WORLD frame, single agentview cam == the finetune-data packer's convention). The npz is
# consumed by pw_flow_render_gif.py (GIF) and pw_ft_grasp_viser.py (interactive viser). This is the
# S0 sanity-viz: SEE the finetuned model move the grasped object, on multiple episodes.
import argparse, glob, os, sys
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim-glob", default="/workspace/tingting/.tmp/s0_sim/*_demo_*.npz")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/s0_ftviz")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cams", default="agentview")  # comma list; single = packer convention
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cams = tuple(args.cams.split(","))

    npzs = sorted(glob.glob(args.sim_glob))[:args.n]
    builder = LiberoDataDictBuilder(domain="droid", device="cuda")
    teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")
    print(f"[ft-imagine] {len(npzs)} episodes, cams={cams}, frame=world", flush=True)

    for npz in npzs:
        base = os.path.basename(npz)[:-4]
        demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
        sr = np.load(npz, allow_pickle=True)
        names = list(sr["obj_names"]); op = sr["obj_poses"]
        cand = [n for n in names if not n.startswith(("robot0", "gripper", "mount"))
                and n not in ("floor", "table", "world")]
        disps = {n: np.linalg.norm(op[-1][names.index(n)][:3, 3] - op[0][names.index(n)][:3, 3]) for n in cand}
        obj = max(disps, key=disps.get); gt = disps[obj]

        H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
        with h5py.File(H, "r") as f:
            o = f["data"][f"demo_{demo}"]["obs"]
            joints = np.asarray(o["joint_states"][:], np.float64)
            grip = np.asarray(o["gripper_states"][:], np.float64)
        dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=cams, frame="world")
        pred = teacher.imagine_from_datadict(dd)["scene_flows"].numpy()   # [T,Ns,3] world

        # near-object pred motion (world frame; matches gate p90)
        opos_w = op[0][names.index(obj)][:3, 3]
        near = np.linalg.norm(pred[0] - opos_w, axis=-1) < 0.08
        pw_mean = np.linalg.norm(pred[-1][near] - pred[0][near], axis=-1).mean() if near.sum() else 0.0
        pw_p90 = np.percentile(np.linalg.norm(pred[-1][near] - pred[0][near], axis=-1), 90) if near.sum() else 0.0

        out = os.path.join(args.out_dir, f"pred_{task}_demo_{demo}.npz")
        np.savez(out,
                 __pred_scene_flows__=pred.astype(np.float32),
                 _scene_colors_u8=dd["_scene_colors_u8"],
                 robot_flows=dd["robot_flows"].astype(np.float32),
                 cam0_intrinsic=dd["cam0_intrinsic"].astype(np.float64),
                 cam0_extrinsic=dd["cam0_extrinsic"].astype(np.float64),
                 obj_name=obj, obj_traj_world=op[:, names.index(obj), :3, 3].astype(np.float32),
                 gt_disp=np.float32(gt))
        print(f"  {task} demo_{demo}: obj={obj} GT={gt:.3f}m near_pred mean={pw_mean:.3f} p90={pw_p90:.3f} -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
