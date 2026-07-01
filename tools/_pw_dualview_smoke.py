# [LangPointWorld] Dual-view smoke: agentview+wrist together through VGGT (shared frame) ->
# merged cloud -> PW imagine. Saves the sample for viz. Compares cloud size vs single-view.
import numpy as np, h5py, sys
from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

HDF5 = sys.argv[1]; CKPT = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "/workspace/tingting/.tmp/dualview_sample_ep0.npz"
T_MODEL = 11
print("[dual] VGGT dual-view t0 ...", flush=True)
g = VGGTGeometryProvider(device="cuda")
dual = g.dual_view_t0(HDF5, "demo_0")
for ck, v in dual.items():
    print(f"  {ck}: depth{v['depth'].shape} K fx={v['K'][0,0]:.1f} E_t={v['E'][:3,3].round(3)}", flush=True)
with h5py.File(HDF5, "r") as f:
    obs = f["data"]["demo_0"]["obs"]
    joints = np.asarray(obs["joint_states"][:], np.float64)
    grip = np.asarray(obs["gripper_states"][:], np.float64)
print("[dual] build dual-view data_dict ...", flush=True)
b = LiberoDataDictBuilder(domain="droid", device="cuda")
dd = b.build_dualview(dual, joints, grip, horizon=T_MODEL)
print(f"[dual] merged scene cloud: {dd['scene_flows'].shape} (single-view was 8192)", flush=True)
print(f"[dual] cameras: {[k for k in dd if k.endswith('_initial_rgb')]}", flush=True)
print("[dual] teacher imagine ...", flush=True)
t = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")
out = t.imagine_from_datadict(dd)
sf = out["scene_flows"]
disp = np.linalg.norm(sf.numpy()[-1] - sf.numpy()[0], axis=-1)
print(f"=== DUAL-VIEW IMAGINE OK === scene_flows {tuple(sf.shape)} finite {bool(sf.isfinite().all())} "
      f"moved>1cm {(disp>0.01).mean():.1%}", flush=True)
# save for viz (same keys pw_official_viz / gif tools expect)
savez = {k: v for k, v in dd.items() if isinstance(v, np.ndarray)}
savez["__pred_scene_flows__"] = sf.numpy()
savez["scene_colors"] = dd["_scene_colors_u8"]
np.savez(OUT, **savez)
print(f"[dual] saved -> {OUT}", flush=True)
