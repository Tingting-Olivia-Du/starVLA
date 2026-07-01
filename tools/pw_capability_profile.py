# [LangPointWorld] Profile WHAT PW gets right/wrong on LIBERO, to gauge finetune difficulty.
# For each episode, split scene points into 3 regions and compare PW's predicted motion to the
# relevant GT signal:
#   (a) near-arm points   -> should follow the arm; compare to the arm's own GT motion (robot_flows
#                            is the GT action input). Tests if PW's dynamics prior fires at all.
#   (b) near-object points -> should follow the grasped object; compare to object GT displacement.
#   (c) static-background   -> should stay still; PW≈0 is correct.
# If (a) is good but (b) is bad => PW dynamics work, it only lacks grasp/contact causality => easy-ish
# finetune. If (a) is also bad => PW dynamics broke on LIBERO => harder.
import glob, os, sys
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

SIM_DIR = sys.argv[1]; CKPT = sys.argv[2]
DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T = 11
builder = LiberoDataDictBuilder(domain="droid", device="cuda")
teacher = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")

def cos_dir(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0

print(f"\n{'task':13s} | {'ARM-region':^22s} | {'OBJ-region':^22s} | {'STATIC bg':^12s}", flush=True)
print(f"{'':13s} | {'GTmove PWpred cos':^22s} | {'GTmove PWpred ratio':^22s} | {'PWpred(≈0 good)':^12s}", flush=True)
print("-" * 78, flush=True)
for npz in sorted(glob.glob(os.path.join(SIM_DIR, "sim_*.npz"))):
    task = os.path.basename(npz)[4:-4]
    sr = np.load(npz, allow_pickle=True)
    names = list(sr["obj_names"]); op = sr["obj_poses"]
    base_T = op[0][names.index("robot0_base")]; base_inv = np.linalg.inv(base_T)
    def to_base(pw): return (base_inv[:3, :3] @ pw) + base_inv[:3, 3]
    with h5py.File(f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5", "r") as f:
        o = f["data"]["demo_0"]["obs"]
        joints = np.asarray(o["joint_states"][:], np.float64); grip = np.asarray(o["gripper_states"][:], np.float64)
    dd = builder.build_from_sim(sr, joints, grip, horizon=T)
    out = teacher.imagine_from_datadict(dd)
    sf = out["scene_flows"].numpy()             # [T,Ns,3] base
    rf = dd["robot_flows"]                       # [T,Nr,3] base (GT arm)

    # (a) arm region: scene points near the arm at t=0; arm GT motion = eef/gripper displacement
    eef_b = to_base(op[0][names.index("gripper0_eef")][:3, 3])
    eef_end_b = to_base(op[-1][names.index("gripper0_eef")][:3, 3])
    arm_gt = np.linalg.norm(eef_end_b - eef_b)
    arm_gt_vec = eef_end_b - eef_b
    near_arm = np.linalg.norm(sf[0] - rf[0].mean(0), axis=-1) < 0.15
    arm_pw = np.linalg.norm(sf[-1][near_arm] - sf[0][near_arm], axis=-1).mean() if near_arm.sum() else 0
    arm_pw_vec = (sf[-1][near_arm] - sf[0][near_arm]).mean(0) if near_arm.sum() else np.zeros(3)
    arm_cos = cos_dir(arm_pw_vec, arm_gt_vec)

    # (b) object region
    obj = max([n for n in names if not n.startswith(("robot0","gripper","mount")) and n not in ("floor","table","world")],
              key=lambda n: np.linalg.norm(op[-1][names.index(n)][:3,3]-op[0][names.index(n)][:3,3]))
    obj_gt = np.linalg.norm(op[-1][names.index(obj)][:3,3]-op[0][names.index(obj)][:3,3])
    obj_b = to_base(op[0][names.index(obj)][:3,3])
    near_obj = np.linalg.norm(sf[0]-obj_b, axis=-1) < 0.08
    obj_pw = np.linalg.norm(sf[-1][near_obj]-sf[0][near_obj], axis=-1).mean() if near_obj.sum() else 0

    # (c) static background: far from both arm and object
    static = (~near_arm) & (~near_obj)
    stat_pw = np.linalg.norm(sf[-1][static]-sf[0][static], axis=-1).mean() if static.sum() else 0

    print(f"{task:13s} | {arm_gt:6.3f} {arm_pw:6.3f} cos{arm_cos:+.2f} | "
          f"{obj_gt:6.3f} {obj_pw:6.3f} r{obj_pw/max(obj_gt,1e-3):.2f} | {stat_pw:12.4f}", flush=True)

print("-" * 78, flush=True)
print("[profile] ARM cos>0 & ARM PWmove>0 => PW dynamics fire near the arm (good).", flush=True)
print("[profile] OBJ ratio<<1 => misses grasped object. STATIC≈0 => correct on static.", flush=True)
print("[profile] => finetune should focus on grasp/contact causality, not rebuild dynamics.", flush=True)
