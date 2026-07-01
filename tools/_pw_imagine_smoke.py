# [LangPointWorld] Standalone imagine smoke: real LIBERO -> VGGT -> teacher -> flow.
import numpy as np, h5py, sys
from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

HDF5 = sys.argv[1]
CKPT = sys.argv[2]
print("[smoke] loading VGGT ...", flush=True)
g = VGGTGeometryProvider(device="cuda")
dp = g.depth_provider(HDF5, "demo_0")
K, E = g.intrinsics_extrinsics(HDF5, "demo_0", 0)
s = build_pw_sample(HDF5, "demo_0", 0, depth_provider=dp, domain="droid", intrinsics=K, extrinsics=E)
print("[smoke] sample keys:", sorted(s.keys()), flush=True)
with h5py.File(HDF5, "r") as f:
    actions = np.asarray(f["data"]["demo_0"]["actions"][0:8], np.float64)
print("[smoke] loading teacher ...", flush=True)
t = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")
print("[smoke] running imagine ...", flush=True)
out = t.imagine(s, demo_action_chunk=actions)
sf = out["scene_flows"]
print("=== IMAGINE OK ===", "scene_flows", tuple(sf.shape), "finite", bool(sf.isfinite().all()),
      "coord0", tuple(out["scene_coord0"].shape), flush=True)
