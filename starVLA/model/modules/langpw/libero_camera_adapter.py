# [LangPointWorld] LIBERO HDF5 episode -> PointWorld raw sample dict (C1, spec §3.1).
# Resolves the depth gap (LIBERO obs has no depth), un-flips agentview, resizes
# 128x128 -> PointWorld's (180,320) contract, converts euler ee_ori -> quaternion.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import numpy as np
import h5py
import cv2

_PW_HW = (180, 320)  # PointWorld release camera-payload resolution (pipeline assert)
_PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]  # domain='droid' Franka layout


def _euler_xyz_to_quat(e):
    # LIBERO ee_ori is an XYZ euler triple. Convert to (x,y,z,w) unit quat.
    e = np.asarray(e, np.float64)
    cx, cy, cz = np.cos(e * 0.5)
    sx, sy, sz = np.sin(e * 0.5)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    q = np.array([x, y, z, w], np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _prep_rgb(img):
    up = np.flipud(np.asarray(img))            # LIBERO agentview stored upside-down
    return cv2.resize(up, (_PW_HW[1], _PW_HW[0]), interpolation=cv2.INTER_AREA)


def _prep_depth(d):
    return cv2.resize(np.asarray(d, np.float32), (_PW_HW[1], _PW_HW[0]),
                      interpolation=cv2.INTER_NEAREST)


def build_pw_sample(hdf5_path, demo, t, *, depth_provider, domain, intrinsics, extrinsics):
    """Build one PointWorld raw sample dict from a LIBERO HDF5 timestep.

    depth_provider(cam_key, t) -> HxW float32 depth (metric). Resolves the
    LIBERO-has-no-depth gap (Phase-0 decision: sim depth / Depth-Anything-3 / VGGT).
    intrinsics/extrinsics: dict cam_key -> 3x3 / 4x4.
    """
    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo]["obs"]
        rgb_a = np.asarray(obs["agentview_rgb"][t])
        rgb_w = np.asarray(obs["eye_in_hand_rgb"][t])
        ee_pos = np.asarray(obs["ee_pos"][t], np.float64)
        ee_ori = np.asarray(obs["ee_ori"][t], np.float64)
        joints = np.asarray(obs["joint_states"][t], np.float64)
    ee_quat = _euler_xyz_to_quat(ee_ori)
    sample = {
        "agentview_initial_rgb": _prep_rgb(rgb_a),
        "agentview_initial_depth": _prep_depth(depth_provider("agentview_rgb", t)),
        "agentview_intrinsic": np.asarray(intrinsics["agentview_rgb"], np.float64),
        "agentview_extrinsic": np.asarray(extrinsics["agentview_rgb"], np.float64),
        "eye_in_hand_initial_rgb": _prep_rgb(rgb_w),
        "eye_in_hand_initial_depth": _prep_depth(depth_provider("eye_in_hand_rgb", t)),
        "eye_in_hand_intrinsic": np.asarray(intrinsics["eye_in_hand_rgb"], np.float64),
        "eye_in_hand_extrinsic": np.asarray(extrinsics["eye_in_hand_rgb"], np.float64),
        "joint_positions": joints,
        "joint_names": list(_PANDA_JOINT_NAMES),
        "base_pose": np.eye(4, dtype=np.float64),
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "__domain__": domain,
    }
    return sample
