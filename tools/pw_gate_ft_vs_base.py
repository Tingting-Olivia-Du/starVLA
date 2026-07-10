# [LangPointWorld] S0-e GATE: does finetuning fix PW's systematic grasped-object under-prediction?
# Runs BOTH the pretrained (base) and finetuned PW on the SAME LIBERO sim clips, and for the
# manipulated object reports GT displacement vs PW-predicted motion of points near it. The v3
# thesis (sample-efficient domain adaptation) passes iff finetuned ratio >> base ratio (base ~0.05).
#
# Usage: python pw_gate_ft_vs_base.py <sim_glob> <base_ckpt> <ft_ckpt> [n_clips]
import glob, os, sys
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

SIM_GLOB = sys.argv[1]
BASE_CKPT = sys.argv[2]
FT_CKPT = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11
NEAR = 0.08

npzs = sorted(glob.glob(SIM_GLOB))[:N]
print(f"[gate] {len(npzs)} clips | base={os.path.basename(os.path.dirname(BASE_CKPT))} ft={os.path.basename(os.path.dirname(FT_CKPT))}", flush=True)

builder = LiberoDataDictBuilder(domain="droid", device="cuda")

def robot_disp_and_gt(sr):
    names = list(sr["obj_names"]); op = sr["obj_poses"]
    cand = [n for n in names if not n.startswith(("robot0", "gripper", "mount"))
            and n not in ("floor", "table", "world")]
    disps = {n: np.linalg.norm(op[-1][names.index(n)][:3, 3] - op[0][names.index(n)][:3, 3]) for n in cand}
    obj = max(disps, key=disps.get)
    return names, op, obj, disps[obj]

def pred_near_obj(sf, sr, names, op, obj):
    # scene is in WORLD frame (matches build_from_sim frame="world"); object center in world too.
    # Report both the ball MEAN (diluted by nearby static/table points) and the ball 90th-pctile
    # displacement (isolates the moving object points from static dilution, == the WDS-native
    # scene_moved_mask quantity the training-faithful probe measured).
    opos_w = op[0][names.index(obj)][:3, 3]
    near = np.linalg.norm(sf[0] - opos_w, axis=-1) < NEAR
    if not near.sum():
        return 0.0, 0.0
    disp = np.linalg.norm(sf[-1][near] - sf[0][near], axis=-1)
    return float(disp.mean()), float(np.percentile(disp, 90))

def run_ckpt(ckpt, tag):
    teacher = PointWorldTeacher(ckpt, ptv3_size="large", domain="droid", device="cuda")
    out_rows = []
    for npz in npzs:
        base = os.path.basename(npz)[:-4]
        demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
        sr = np.load(npz, allow_pickle=True)
        names, op, obj, gt = robot_disp_and_gt(sr)
        H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
        with h5py.File(H, "r") as f:
            o = f["data"][f"demo_{demo}"]["obs"]
            joints = np.asarray(o["joint_states"][:], np.float64)
            grip = np.asarray(o["gripper_states"][:], np.float64)
        # WORLD frame + single agentview cam == exactly the finetune-data packer's convention
        dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=("agentview",), frame="world")
        sf = teacher.imagine_from_datadict(dd)["scene_flows"].numpy()
        mean_d, p90_d = pred_near_obj(sf, sr, names, op, obj)
        out_rows.append((task, demo, obj, gt, mean_d, p90_d))
    del teacher
    import torch; torch.cuda.empty_cache()
    return out_rows

base_rows = run_ckpt(BASE_CKPT, "base")
ft_rows = run_ckpt(FT_CKPT, "ft")

print(f"\n{'task':14s} {'demo':>4s} {'object':18s} {'GT':>7s} {'b_p90':>6s} {'br':>5s} {'ft_p90':>6s} {'fr':>5s}", flush=True)
print("-" * 74, flush=True)
for b, f in zip(base_rows, ft_rows):
    print(f"{b[0][:14]:14s} {b[1]:>4s} {b[2][:18]:18s} {b[3]:7.3f} {b[5]:6.3f} {b[5]/max(b[3],1e-3):5.2f} {f[5]:6.3f} {f[5]/max(f[3],1e-3):5.2f}", flush=True)
print("-" * 74, flush=True)
bg = np.array([r[3] for r in base_rows])
# p90 = moving-object displacement (isolated from static dilution); mean = ball-average
b_p90 = np.array([r[5] for r in base_rows]); f_p90 = np.array([r[5] for r in ft_rows])
b_mean = np.array([r[4] for r in base_rows]); f_mean = np.array([r[4] for r in ft_rows])
br = (b_p90 / np.clip(bg, 1e-3, None)).mean(); fr = (f_p90 / np.clip(bg, 1e-3, None)).mean()
br_m = (b_mean / np.clip(bg, 1e-3, None)).mean(); fr_m = (f_mean / np.clip(bg, 1e-3, None)).mean()
print(f"[gate] ball-MEAN : BASE {br_m*100:5.1f}% | FT {fr_m*100:5.1f}%", flush=True)
print(f"[gate] obj-P90   : BASE {br*100:5.1f}% | FT {fr*100:5.1f}%  (moving-object motion, static-diluted removed)", flush=True)
print(f"[gate] improvement (p90): {fr/max(br,1e-3):.1f}x", flush=True)
print(f"[gate] {'PASS' if fr > 0.5 else 'PARTIAL' if fr > 2*br else 'FAIL'}: FT p90 ratio {fr:.2f} (target >0.5, base {br:.2f})", flush=True)
