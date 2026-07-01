# [LangPointWorld] Batch grasp-miss check across LIBERO episodes: load PW teacher ONCE, then for
# each sim-render npz build the aligned data_dict, imagine, and report the manipulated object's GT
# displacement vs PW's predicted motion near it. Confirms whether PW-misses-grasp is universal.
import glob, os, sys
import numpy as np
from scipy.spatial import cKDTree
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

SIM_DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace/tingting/.tmp/multiep"
CKPT = sys.argv[2]
DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11

builder = LiberoDataDictBuilder(domain="droid", device="cuda")
teacher = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")

print(f"\n{'task':16s} {'object':22s} {'GT_disp':>8s} {'PW_pred':>8s} {'ratio':>6s}", flush=True)
print("-" * 66, flush=True)
rows = []
for npz in sorted(glob.glob(os.path.join(SIM_DIR, "sim_*.npz"))):
    task = os.path.basename(npz)[4:-4]
    sr = np.load(npz, allow_pickle=True)
    names = list(sr["obj_names"])
    op = sr["obj_poses"]
    # manipulated object = the task object with max GT displacement (exclude robot/gripper/static)
    cand = [n for n in names if not n.startswith(("robot0", "gripper", "mount"))
            and n not in ("floor", "table", "world")]
    disps = {n: np.linalg.norm(op[-1][names.index(n)][:3, 3] - op[0][names.index(n)][:3, 3]) for n in cand}
    obj = max(disps, key=disps.get)
    gt = disps[obj]

    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
    with h5py.File(H, "r") as f:
        o = f["data"]["demo_0"]["obs"]
        joints = np.asarray(o["joint_states"][:], np.float64)
        grip = np.asarray(o["gripper_states"][:], np.float64)
    dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL)
    out = teacher.imagine_from_datadict(dd)
    sf = out["scene_flows"].numpy()
    base_T = op[0][names.index("robot0_base")]; base_inv = np.linalg.inv(base_T)
    opos_w = op[0][names.index(obj)][:3, 3]
    opos_b = base_inv[:3, :3] @ opos_w + base_inv[:3, 3]
    near = np.linalg.norm(sf[0] - opos_b, axis=-1) < 0.08
    pw = np.linalg.norm(sf[-1][near] - sf[0][near], axis=-1).mean() if near.sum() else 0.0
    ratio = pw / gt if gt > 1e-3 else 0
    rows.append((task, obj, gt, pw, ratio))
    print(f"{task:16s} {obj[:22]:22s} {gt:8.3f} {pw:8.3f} {ratio:6.2f}", flush=True)

print("-" * 66, flush=True)
gts = np.array([r[2] for r in rows]); pws = np.array([r[3] for r in rows])
print(f"{'MEAN':16s} {'':22s} {gts.mean():8.3f} {pws.mean():8.3f} {(pws/np.clip(gts,1e-3,None)).mean():6.2f}", flush=True)
print(f"\n[batch] PW predicts {(pws/np.clip(gts,1e-3,None)).mean()*100:.1f}% of the object's true motion on average.", flush=True)
print("[batch] ratio<<1 across tasks => PW systematically misses grasped-object motion on LIBERO.", flush=True)
