# [LangPointWorld] Full imagine smoke: LIBERO -> VGGT -> bridge data_dict -> teacher -> flow.
import numpy as np, h5py, sys
from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

HDF5 = sys.argv[1]
CKPT = sys.argv[2]
HORIZON = 11
print("[smoke] VGGT ...", flush=True)
g = VGGTGeometryProvider(device="cuda")
dp = g.depth_provider(HDF5, "demo_0")
K, E = g.intrinsics_extrinsics(HDF5, "demo_0", 0)
s = build_pw_sample(HDF5, "demo_0", 0, depth_provider=dp, domain="droid", intrinsics=K, extrinsics=E)
with h5py.File(HDF5, "r") as f:
    obs = f["data"]["demo_0"]["obs"]
    joints = np.asarray(obs["joint_states"][:], np.float64)
    grip = np.asarray(obs["gripper_states"][:], np.float64)
print("[smoke] bridge (FK + unproject) ...", flush=True)
b = LiberoDataDictBuilder(domain="droid", device="cuda")
dd = b.build(s, joints, grip, horizon=HORIZON)
print("[smoke] data_dict scene_flows", dd["scene_flows"].shape, "robot_flows", dd["robot_flows"].shape, flush=True)
print("[smoke] teacher ...", flush=True)
t = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")
print("[smoke] imagine ...", flush=True)
out = t.imagine_from_datadict(dd)
sf = out["scene_flows"]
print("=== IMAGINE OK ===", "scene_flows", tuple(sf.shape), "finite", bool(sf.isfinite().all()), flush=True)
# save for inspection
np.savez("/workspace/tingting/.tmp/imagine_out.npz",
         scene_flows=sf.numpy(), scene_coord0=out["scene_coord0"].numpy())
print("[smoke] saved imagine_out.npz", flush=True)
