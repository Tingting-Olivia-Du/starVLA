# [LangPointWorld] LIBERO -> PointWorld data_dict bridge (the real integration seam).
#
# KEY ARCHITECTURAL FINDING (see preflight note §7): PointWorld's `main` branch does NOT
# unproject RGB-D. It consumes data-branch-preprocessed `scene_flows` (GT full-scene point
# TRAJECTORIES from mesh+pose tracking) + `robot_flows` (FK over joints). For the PW-PROBE we
# reconstruct the *minimal* data_dict BaseModel.forward consumes:
#   - scene_flows[:,0]  = t=0 3D scene point cloud, obtained by unprojecting VGGT depth (K,E).
#     Future frames are dummies — the model PREDICTS them (that is the imagined output).
#   - robot_flows[T,Nr,3] = FK over the demo joint trajectory via PointWorld's RobotSampler.
#     This is the ACTION representation the teacher conditions on.
# This bypasses the WDS `apply_release_pipeline_to_sample` (data-branch-only) entirely.
#
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import os
import sys
import numpy as np
import torch

_PW_ROOT = "/workspace/tingting/PointWorld"
_PW_EXTRA_SITE = "/workspace/tingting/envs/pw_extra_site"


def _ensure_pw_on_path():
    for p in (_PW_EXTRA_SITE, _PW_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def unproject_depth_to_points(depth, K, E_cam_from_world, rgb=None, max_points=None, rng=None):
    """depth[H,W] metric, K[3,3], E[4,4] (camera-from-world), rgb[H,W,3] optional.
    Returns (world-frame points [N,3], colors [N,3] in 0..1, (rows,cols)). Backproject valid
    depth pixels via K^-1 then E^-1 (E maps world->camera)."""
    H, W = depth.shape
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 1e-4))
    z = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)                 # [N,3] camera frame
    E_inv = np.linalg.inv(E_cam_from_world)
    ones = np.ones((pts_cam.shape[0], 1), pts_cam.dtype)
    pts_world = (E_inv @ np.concatenate([pts_cam, ones], -1).T).T[:, :3]
    if max_points is not None and pts_world.shape[0] > max_points:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(pts_world.shape[0], max_points, replace=False)
        pts_world = pts_world[sel]; ys, xs = ys[sel], xs[sel]
    colors = (rgb[ys, xs].astype(np.float32) / 255.0) if rgb is not None else np.zeros_like(pts_world)
    return pts_world.astype(np.float32), colors, (ys, xs)


class LiberoDataDictBuilder:
    """Builds the minimal data_dict BaseModel.forward consumes, from a LIBERO sample +
    VGGT geometry + a PointWorld RobotSampler (droid Franka URDF)."""

    def __init__(self, domain="droid", device="cuda", max_scene_points=8192, max_robot_points=512):
        _ensure_pw_on_path()
        self._prev_cwd = os.getcwd()
        os.chdir(_PW_ROOT)  # droid URDF path is relative to repo root
        from robot_sampler import RobotSampler
        from utils import resolve_robot_urdf
        self.domain = domain
        self.device = device
        self.max_scene_points = max_scene_points
        self.max_robot_points = max_robot_points
        urdf = resolve_robot_urdf(domain)
        self.robot_sampler = RobotSampler(urdf, device=device)

    def build_robot_flows(self, joint_states, gripper_states):
        """joint_states[T,7], gripper_states[T,2] -> robot_flows[T,Nr,3] (world frame)."""
        _ensure_pw_on_path()
        from dataset_components.robot import _get_robot_flows_droid
        # LIBERO gripper_states[T,2] -> a single finger position proxy (mean of the two jaws).
        gripper_pos = np.asarray(gripper_states, np.float64).mean(axis=1)  # [T]
        sample = {
            "joint_positions": np.asarray(joint_states, np.float64),
            "gripper_positions": gripper_pos,
            "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        }
        out = _get_robot_flows_droid(sample, self.robot_sampler, self.max_robot_points)
        return np.asarray(out["robot_flows"], np.float32)  # [T,Nr,3]

    def build(self, libero_sample, joint_states, gripper_states, horizon):
        """libero_sample: build_pw_sample output (RGB/depth/K/E per cam). Returns a data_dict
        with numpy arrays ready to be tensor-ized + unsqueezed by the teacher. Reuses
        PointWorld's gather_features so scene/robot features (31/16-dim) are REAL, not fabricated
        — otherwise the probe would measure the teacher on garbage features."""
        _ensure_pw_on_path()
        from dataset_components.robot import gather_features
        # --- scene points at t=0 from agentview depth unprojection (world frame) ---
        depth = libero_sample["agentview_initial_depth"]        # [180,320] (already PW res)
        rgb = libero_sample["agentview_initial_rgb"]            # [180,320,3]
        K = libero_sample["agentview_intrinsic"].copy()
        E = libero_sample["agentview_extrinsic"]
        H0, W0 = 128, 128
        H1, W1 = depth.shape
        K[0, :] *= (W1 / W0); K[1, :] *= (H1 / H0)   # native-res K -> resized payload
        scene0, scene_colors0, _ = unproject_depth_to_points(
            depth, K, E, rgb=rgb, max_points=self.max_scene_points)
        Ns = scene0.shape[0]
        T = horizon
        scene_flows = np.tile(scene0[None], (T, 1, 1)).astype(np.float32)  # static-scene init; model predicts motion
        scene_exists = np.ones((T, Ns), bool)
        scene_colors = np.tile(scene_colors0[None], (T, 1, 1)).astype(np.float32)
        scene_normals = np.zeros((T, Ns, 3), np.float32)      # normals unavailable from mono depth -> zeros

        rf_out = self._robot_flows_full(joint_states[:T], gripper_states[:T], T)
        robot_flows = rf_out["robot_flows"]; robot_colors = rf_out["robot_colors"]; robot_normals = rf_out["robot_normals"]
        Nr = robot_flows.shape[1]
        robot_exists = np.ones((T, Nr), bool)

        # gripper_open from LIBERO gripper_states (mean jaw -> open scalar in 0..1)
        gopen = np.asarray(gripper_states[:T], np.float64).mean(axis=1).reshape(T, 1).astype(np.float32)
        if gopen.shape[0] < T:
            gopen = np.concatenate([gopen, np.repeat(gopen[-1:], T - gopen.shape[0], 0)], 0)

        feat_sample = {
            "scene_flows": scene_flows, "scene_colors": scene_colors, "scene_normals": scene_normals,
            "robot_flows": robot_flows, "robot_colors": robot_colors, "robot_normals": robot_normals,
            "right_gripper_open": gopen,
        }
        gather_features(feat_sample, domain=self.domain)
        scene_features_t0 = feat_sample["scene_features"]        # (1 or T, Ns, 31)
        if scene_features_t0.shape[0] == 1:
            scene_features = np.tile(scene_features_t0, (T, 1, 1))
        else:
            scene_features = scene_features_t0
        robot_features = feat_sample["robot_features"]           # (T, Nr, 16)

        data_dict = {
            "scene_flows": scene_flows.astype(np.float32),
            "scene_features": scene_features.astype(np.float32),
            "scene_exists": scene_exists,
            "robot_flows": robot_flows.astype(np.float32),
            "robot_features": robot_features.astype(np.float32),
            "robot_exists": robot_exists,
            "cam0_initial_rgb": rgb,
            "cam0_initial_depth": depth.astype(np.float32),
            "cam0_intrinsic": K.astype(np.float64),
            "cam0_extrinsic": E.astype(np.float64),
            "__domain__": self.domain,
        }
        return data_dict

    def _robot_flows_full(self, joint_states, gripper_states, T):
        """Like build_robot_flows but returns colors/normals too, padded to T frames."""
        _ensure_pw_on_path()
        from dataset_components.robot import _get_robot_flows_droid
        gripper_pos = np.asarray(gripper_states, np.float64).mean(axis=1)
        sample = {
            "joint_positions": np.asarray(joint_states, np.float64),
            "gripper_positions": gripper_pos,
            "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        }
        out = _get_robot_flows_droid(sample, self.robot_sampler, self.max_robot_points)
        rf = np.asarray(out["robot_flows"], np.float32)
        rc = np.asarray(out["robot_colors"], np.float32)
        rn = np.asarray(out["robot_normals"], np.float32)
        def _pad(a):
            return a if a.shape[0] >= T else np.concatenate([a, np.repeat(a[-1:], T - a.shape[0], 0)], 0)
        return {"robot_flows": _pad(rf), "robot_colors": _pad(rc), "robot_normals": _pad(rn)}

    def __del__(self):
        try:
            os.chdir(self._prev_cwd)
        except Exception:
            pass
