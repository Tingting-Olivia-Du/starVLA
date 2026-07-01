# tests/geomemvla/test_pw_camera_adapter.py
# [LangPointWorld] Contract test: LIBERO episode -> PointWorld raw sample dict (CPU).
import numpy as np
import h5py
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample


def _make_synth_hdf5(path, T=4):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0/obs")
        g.create_dataset("agentview_rgb", data=np.zeros((T, 128, 128, 3), np.uint8))
        g.create_dataset("eye_in_hand_rgb", data=np.zeros((T, 128, 128, 3), np.uint8))
        g.create_dataset("ee_pos", data=np.zeros((T, 3), np.float64))
        g.create_dataset("ee_ori", data=np.zeros((T, 3), np.float64))
        g.create_dataset("joint_states", data=np.zeros((T, 7), np.float64))
        g.create_dataset("gripper_states", data=np.zeros((T, 2), np.float64))
        f.create_dataset("data/demo_0/actions", data=np.zeros((T, 7), np.float64))


def test_sample_has_pointworld_contract_keys(tmp_path):
    p = str(tmp_path / "e.hdf5")
    _make_synth_hdf5(p)
    depth = lambda cam, t: np.ones((128, 128), np.float32)
    K = {"agentview_rgb": np.eye(3), "eye_in_hand_rgb": np.eye(3)}
    E = {"agentview_rgb": np.eye(4), "eye_in_hand_rgb": np.eye(4)}
    s = build_pw_sample(p, "demo_0", 0, depth_provider=depth, domain="droid", intrinsics=K, extrinsics=E)
    for k in ["agentview_initial_rgb", "agentview_initial_depth", "agentview_intrinsic",
              "agentview_extrinsic", "joint_positions", "joint_names", "base_pose",
              "ee_pos", "ee_quat"]:
        assert k in s, f"missing {k}"
    assert s["agentview_initial_rgb"].shape[:2] == (180, 320), "RGB must be resized to (180,320)"
    assert s["agentview_initial_depth"].shape[:2] == (180, 320)
    assert len(s["joint_positions"]) == 7
    assert s["ee_quat"].shape[-1] == 4


def test_agentview_is_unflipped(tmp_path):
    # LIBERO stores agentview upside-down; adapter must vertical-flip so scene is upright.
    p = str(tmp_path / "e.hdf5")
    with h5py.File(p, "w") as f:
        g = f.create_group("data/demo_0/obs")
        img = np.zeros((1, 128, 128, 3), np.uint8)
        img[0, 0, :, :] = 255  # top row white in stored (upside-down) frame
        g.create_dataset("agentview_rgb", data=img)
        g.create_dataset("eye_in_hand_rgb", data=np.zeros((1, 128, 128, 3), np.uint8))
        for k, sh in [("ee_pos", 3), ("ee_ori", 3), ("joint_states", 7), ("gripper_states", 2)]:
            g.create_dataset(k, data=np.zeros((1, sh), np.float64))
        f.create_dataset("data/demo_0/actions", data=np.zeros((1, 7), np.float64))
    depth = lambda cam, t: np.ones((128, 128), np.float32)
    K = {"agentview_rgb": np.eye(3), "eye_in_hand_rgb": np.eye(3)}
    E = {"agentview_rgb": np.eye(4), "eye_in_hand_rgb": np.eye(4)}
    s = build_pw_sample(p, "demo_0", 0, depth_provider=depth, domain="droid", intrinsics=K, extrinsics=E)
    rgb = s["agentview_initial_rgb"]
    assert rgb[-1, :, :].mean() > rgb[0, :, :].mean(), "white row should be at bottom after un-flip"
