# [LangPointWorld] Feed the (denser) dual-view point cloud into the FINETUNED PointWorld and let it
# PREDICT future scene flow. Runs TWO densities on the SAME episode so we can see whether denser
# input helps or hurts (PW trains at ~8192 pts / official enforce_max=10000, so >>10k is OOD):
#   - 8192  (matches S0 finetune / official distribution)
#   - 16384 (denser, what the user asked to feed)
# Reports predicted grasped-object motion (near-object p90 disp) for each, and dumps a pred npz per
# density for the viser (pw_pred_viser.py).
import argparse, glob, os, sys
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11
FT = "/workspace/tingting/.tmp/s0_train/dummy-qbyz7nd4/model-last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="/workspace/tingting/.tmp/s0_sim/alphabet_soup_demo_0.npz")
    ap.add_argument("--ckpt", default=FT)
    ap.add_argument("--densities", default="8192,16384")
    ap.add_argument("--sampling", default="uniform", choices=["uniform", "importance", "both"],
                    help="both = run uniform AND importance at each density for comparison")
    ap.add_argument("--official", default="off", choices=["off", "on", "both"],
                    help="official preprocessing (center+voxel1.5cm+12k+normalize). both=compare off vs on")
    ap.add_argument("--foveated", action="store_true",
                    help="with --official on: keep ALL task-object points full-res, voxel only the rest")
    ap.add_argument("--single", action="store_true", help="single agentview (default DUAL)")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/pw_pred_density")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cams = ("agentview",) if args.single else ("agentview", "robot0_eye_in_hand")
    densities = [int(x) for x in args.densities.split(",")]

    base = os.path.basename(args.sim)[:-4]
    demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
    sr = np.load(args.sim, allow_pickle=True)
    names = list(sr["obj_names"]); op = sr["obj_poses"]
    cand = [n for n in names if not n.startswith(("robot0", "gripper", "mount"))
            and n not in ("floor", "table", "world")]
    disps = {n: np.linalg.norm(op[-1][names.index(n)][:3, 3] - op[0][names.index(n)][:3, 3]) for n in cand}
    obj = max(disps, key=disps.get); gt = disps[obj]; opos = op[0][names.index(obj)][:3, 3]
    with h5py.File(H, "r") as f:
        o = f["data"][f"demo_{demo}"]["obs"]
        joints = np.asarray(o["joint_states"][:], np.float64); grip = np.asarray(o["gripper_states"][:], np.float64)

    teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")
    # variant list: (tag, build kwargs)
    if args.official == "both":
        variants = [("raw", dict(official_preprocess=False)),
                    ("official", dict(official_preprocess=True))]
        if args.foveated:
            variants.append(("official+foveated", dict(official_preprocess=True, official_importance=True)))
    else:
        base_kw = dict(official_preprocess=(args.official == "on"))
        if args.foveated and args.official == "on":
            base_kw["official_importance"] = True
        variants = [("official+foveated" if base_kw.get("official_importance") else
                     ("official" if base_kw["official_preprocess"] else "raw"), base_kw)]
    print(f"[pred] {task} demo_{demo} obj={obj} GT_disp={gt:.3f}m cams={cams}", flush=True)
    print(f"[pred] {'variant':>18s} {'density':>8s} {'Ns':>7s} {'near_p90':>9s} {'ratio_p90':>10s}", flush=True)
    for dens in densities:
        for tag, kw in variants:
            b = LiberoDataDictBuilder(domain="droid", device="cuda", max_scene_points=dens)
            dd = b.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=cams, frame="world",
                                  sampling=args.sampling if args.sampling != "both" else "uniform", **kw)
            pred = teacher.imagine_from_datadict(dd)["scene_flows"].numpy()   # [T,Ns,3]
            shift = dd.get("__shift_amount__", np.zeros(3))
            opos_s = opos + shift
            near = np.linalg.norm(pred[0] - opos_s, axis=-1) < 0.08
            p90 = float(np.percentile(np.linalg.norm(pred[-1][near] - pred[0][near], axis=-1), 90)) if near.sum() else 0.0
            print(f"[pred] {tag:>18s} {dens:>8d} {pred.shape[1]:>7d} {p90:>9.3f} {p90/max(gt,1e-3):>10.2f}", flush=True)
            np.savez(os.path.join(args.out_dir, f"{task}_demo_{demo}_pred_{tag.replace('+','_')}_{dens}.npz"),
                     pred_scene_flows=pred.astype(np.float32),
                     scene_colors=dd["_scene_colors_u8"][:pred.shape[1]],
                     robot_flows=dd["robot_flows"].astype(np.float32),
                     obj_traj_world=(op[:, names.index(obj), :3, 3] + shift).astype(np.float32),
                     gt_disp=np.float32(gt), density=np.int64(dens))
    print(f"[pred] DONE -> {args.out_dir}  (denser helps if its ratio_p90 > the 8192 baseline; "
          f"drops => OOD point count hurt PW)", flush=True)


if __name__ == "__main__":
    main()
