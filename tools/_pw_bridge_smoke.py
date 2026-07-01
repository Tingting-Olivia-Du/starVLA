# [LangPointWorld] Bridge smoke: robot FK + depth unprojection from real LIBERO (no teacher).
import numpy as np, h5py, sys
from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder

HDF5 = sys.argv[1]
print("[bridge] VGGT ...", flush=True)
g = VGGTGeometryProvider(device="cuda")
dp = g.depth_provider(HDF5, "demo_0")
K, E = g.intrinsics_extrinsics(HDF5, "demo_0", 0)
s = build_pw_sample(HDF5, "demo_0", 0, depth_provider=dp, domain="droid", intrinsics=K, extrinsics=E)
with h5py.File(HDF5, "r") as f:
    obs = f["data"]["demo_0"]["obs"]
    joints = np.asarray(obs["joint_states"][:], np.float64)
    grip = np.asarray(obs["gripper_states"][:], np.float64)
print("[bridge] building RobotSampler + FK (loads Franka URDF) ...", flush=True)
b = LiberoDataDictBuilder(domain="droid", device="cuda")
rf = b.build_robot_flows(joints[:8], grip[:8])
print("[bridge] robot_flows:", rf.shape, "finite", bool(np.isfinite(rf).all()), flush=True)
dd = b.build(s, joints, grip, horizon=8)
print("[bridge] scene_flows:", dd["scene_flows"].shape, "t0 nonzero pts:",
      int((np.abs(dd["scene_flows"][0]).sum(-1) > 0).sum()), flush=True)
print("=== BRIDGE OK ===", flush=True)
