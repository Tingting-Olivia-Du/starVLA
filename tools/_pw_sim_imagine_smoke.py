# [LangPointWorld] Sim-aligned imagine: sim real depth/K/E (world frame) -> PW imagine.
# Runs in the starVLA env (uses the pre-rendered sim npz from the libero env).
import numpy as np, h5py, sys
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

SIM_NPZ = sys.argv[1]; HDF5 = sys.argv[2]; CKPT = sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else "/workspace/tingting/.tmp/sim_imagine_ep0.npz"
T_MODEL = 11
print("[sim-imagine] load sim npz + build world-aligned data_dict ...", flush=True)
z = np.load(SIM_NPZ, allow_pickle=True)
with h5py.File(HDF5, "r") as f:
    obs = f["data"]["demo_0"]["obs"]
    joints = np.asarray(obs["joint_states"][:], np.float64)
    grip = np.asarray(obs["gripper_states"][:], np.float64)
b = LiberoDataDictBuilder(domain="droid", device="cuda")
dd = b.build_from_sim(z, joints, grip, horizon=T_MODEL)
print(f"[sim-imagine] scene {dd['scene_flows'].shape} robot {dd['robot_flows'].shape}", flush=True)
sc, rf = dd["scene_flows"][0], dd["robot_flows"][0]
print(f"[sim-imagine] scene z[{sc[:,2].min():.2f},{sc[:,2].max():.2f}] robot z[{rf[:,2].min():.2f},{rf[:,2].max():.2f}]", flush=True)
from scipy.spatial import cKDTree
d, _ = cKDTree(sc).query(rf)
print(f"[sim-imagine] robot-to-nearest-scene dist median {np.median(d):.3f} (VGGT was 0.44)", flush=True)
print("[sim-imagine] teacher imagine ...", flush=True)
t = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")
out = t.imagine_from_datadict(dd)
sf = out["scene_flows"]
disp = np.linalg.norm(sf.numpy()[-1] - sf.numpy()[0], axis=-1)
print(f"=== SIM IMAGINE OK === scene_flows {tuple(sf.shape)} finite {bool(sf.isfinite().all())} "
      f"moved>1cm {(disp>0.01).mean():.1%}", flush=True)
# where does the predicted motion happen relative to the arm? (the key diagnostic)
mov = sf.numpy()[0][disp > 0.01]
if len(mov):
    print(f"[sim-imagine] moving pts z mean {mov[:,2].mean():.3f} vs scene z mean {sc[:,2].mean():.3f} "
          f"(closer to 0 = on table/objects, not floating)", flush=True)
save = {k: v for k, v in dd.items() if isinstance(v, np.ndarray)}
save["__pred_scene_flows__"] = sf.numpy()
save["scene_colors"] = dd["_scene_colors_u8"]
np.savez(OUT, **save)
print(f"[sim-imagine] saved -> {OUT}", flush=True)
