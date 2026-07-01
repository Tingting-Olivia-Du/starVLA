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

    @staticmethod
    def _time_indices(n_frames, horizon, mode="linspace"):
        """Which `horizon` real demo timesteps the T model-steps map to.
        'linspace' (default): equally spaced over the WHOLE demo, so the action covers the full
          approach->grasp->move->place arc (fixes the 'front 7% only' near-static issue).
        'head': the first `horizon` consecutive frames (legacy)."""
        if mode == "head" or n_frames <= horizon:
            idx = np.arange(min(horizon, n_frames))
            if idx.shape[0] < horizon:
                idx = np.concatenate([idx, np.full(horizon - idx.shape[0], idx[-1])])
            return idx
        return np.linspace(0, n_frames - 1, horizon).round().astype(int)

    def build(self, libero_sample, joint_states, gripper_states, horizon, time_mode="linspace"):
        """libero_sample: build_pw_sample output (RGB/depth/K/E per cam). Returns a data_dict
        with numpy arrays ready to be tensor-ized + unsqueezed by the teacher. Reuses
        PointWorld's gather_features so scene/robot features (31/16-dim) are REAL, not fabricated
        — otherwise the probe would measure the teacher on garbage features.
        time_mode selects which demo timesteps the T action steps map to (see _time_indices)."""
        _ensure_pw_on_path()
        from dataset_components.robot import gather_features
        # sample the T action timesteps across the demo (default: full-demo equal spacing)
        ti = self._time_indices(len(joint_states), horizon, mode=time_mode)
        joint_states = np.asarray(joint_states)[ti]
        gripper_states = np.asarray(gripper_states)[ti]
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

        rf_out = self._robot_flows_full(joint_states, gripper_states, T)
        robot_flows = rf_out["robot_flows"]; robot_colors = rf_out["robot_colors"]; robot_normals = rf_out["robot_normals"]
        Nr = robot_flows.shape[1]
        robot_exists = np.ones((T, Nr), bool)

        # gripper_open from LIBERO gripper_states (mean jaw -> open scalar in 0..1)
        gopen = np.asarray(gripper_states, np.float64).mean(axis=1).reshape(-1, 1).astype(np.float32)
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

    def build_dualview(self, dual_geo, joint_states, gripper_states, horizon,
                       time_mode="linspace", cam_keys=("agentview_rgb", "eye_in_hand_rgb")):
        """Dual-view variant: dual_geo = VGGTGeometryProvider.dual_view_t0() output (both cameras
        in ONE shared VGGT frame). Merges both views' t=0 unprojected clouds into one scene point
        cloud AND passes both cameras (cam0=agentview, cam1=wrist) so PW's multi-view scene encoder
        uses both. This is the user's insight: PW natively supports multi-camera scene input; a
        second view yields a more complete, better-triangulated 3D cloud than agentview mono alone."""
        import cv2 as _cv2
        _ensure_pw_on_path()
        from dataset_components.robot import gather_features

        _PW_HW = (180, 320)
        cams_out, cloud_pts, cloud_cols = {}, [], []
        for i, ck in enumerate(cam_keys):
            g = dual_geo[ck]
            pts, cols, _ = unproject_depth_to_points(
                g["depth"], g["K"], g["E"], rgb=g["rgb"],
                max_points=self.max_scene_points // len(cam_keys))
            cloud_pts.append(pts); cloud_cols.append((np.clip(cols, 0, 1) * 255).astype(np.uint8))
            # camera payload for PW scene encoder, resized to (180,320)
            rgb_p = _cv2.resize(g["rgb"], (_PW_HW[1], _PW_HW[0]), interpolation=_cv2.INTER_AREA)
            dep_p = _cv2.resize(g["depth"], (_PW_HW[1], _PW_HW[0]), interpolation=_cv2.INTER_NEAREST)
            Kp = g["K"].copy()
            Kp[0, :] *= (_PW_HW[1] / g["depth"].shape[1]); Kp[1, :] *= (_PW_HW[0] / g["depth"].shape[0])
            cams_out[f"cam{i}"] = {"rgb": rgb_p, "depth": dep_p.astype(np.float32),
                                  "K": Kp.astype(np.float64), "E": g["E"].astype(np.float64)}
        scene0 = np.concatenate(cloud_pts, 0).astype(np.float32)      # merged cloud (shared frame)
        scene_colors0 = np.concatenate(cloud_cols, 0)
        Ns = scene0.shape[0]
        T = horizon
        scene_flows = np.tile(scene0[None], (T, 1, 1))
        scene_exists = np.ones((T, Ns), bool)
        scene_colors = np.tile((scene_colors0.astype(np.float32) / 255.0)[None], (T, 1, 1))
        scene_normals = np.zeros((T, Ns, 3), np.float32)

        # robot flows (same as single-view)
        ti = self._time_indices(len(joint_states), horizon, mode=time_mode)
        js, gs = np.asarray(joint_states)[ti], np.asarray(gripper_states)[ti]
        rf_out = self._robot_flows_full(js, gs, T)
        robot_flows, robot_colors, robot_normals = rf_out["robot_flows"], rf_out["robot_colors"], rf_out["robot_normals"]
        Nr = robot_flows.shape[1]
        robot_exists = np.ones((T, Nr), bool)
        gopen = np.asarray(gs, np.float64).mean(axis=1).reshape(-1, 1).astype(np.float32)

        feat_sample = {
            "scene_flows": scene_flows, "scene_colors": scene_colors, "scene_normals": scene_normals,
            "robot_flows": robot_flows, "robot_colors": robot_colors, "robot_normals": robot_normals,
            "right_gripper_open": gopen,
        }
        gather_features(feat_sample, domain=self.domain)
        sf = feat_sample["scene_features"]
        scene_features = np.tile(sf, (T, 1, 1)) if sf.shape[0] == 1 else sf

        data_dict = {
            "scene_flows": scene_flows.astype(np.float32),
            "scene_features": scene_features.astype(np.float32),
            "scene_exists": scene_exists,
            "scene_supervised_mask": np.ones((T, Ns), dtype=bool),
            "robot_flows": robot_flows.astype(np.float32),
            "robot_features": feat_sample["robot_features"].astype(np.float32),
            "robot_exists": robot_exists,
            "__domain__": self.domain,
            "_scene_colors_u8": scene_colors0,  # for viz
        }
        for cname, c in cams_out.items():
            data_dict[f"{cname}_initial_rgb"] = c["rgb"]
            data_dict[f"{cname}_initial_depth"] = c["depth"]
            data_dict[f"{cname}_intrinsic"] = c["K"]
            data_dict[f"{cname}_extrinsic"] = c["E"]
        return data_dict

    def build_from_sim(self, sim_npz, joint_states, gripper_states, horizon,
                       cams=("agentview", "robot0_eye_in_hand"), flip_depth=True):
        """Build a WORLD-FRAME-ALIGNED data_dict from tools/pw_sim_render.py output. Scene cloud =
        MULTI-VIEW: each camera's sim REAL metric depth unprojected with its sim REAL K/E
        (cam->world) and MERGED (both views share the sim world frame -> aligned). Robot cloud = FK
        points transformed to world via the sim robot0_base pose. Scene+robot in ONE metric world
        frame, zero misalignment. `cams` = the PW views (cam0, cam1, ...). Single-view = one cam."""
        import cv2 as _cv2
        _ensure_pw_on_path()
        from dataset_components.robot import gather_features
        _PW_HW = (180, 320)

        z = sim_npz
        names = list(z["obj_names"])
        base_T = z["obj_poses"][0][names.index("robot0_base")]      # robot base -> world (t=0)
        base_inv = np.linalg.inv(base_T)                            # world -> base

        # Everything is expressed in the ROBOT-BASE frame (base at origin), matching the official
        # viz's URDF mesh which is FK'd from the base origin. Scene: world -> base. Robot: keep FK
        # points (already base frame). Camera extrinsic: base -> cam = (world->cam) @ (base->world).
        cams = tuple(cams) if isinstance(cams, (list, tuple)) else (cams,)
        per_cam = []            # payloads for PW cam0/cam1/...
        cloud_pts, cloud_cols = [], []
        per_cam_budget = max(1, self.max_scene_points // len(cams))
        for c in cams:
            depth = z[f"{c}_depth"][0].copy()
            rgb0 = z[f"{c}_rgb"][0]
            if flip_depth:                                         # robosuite depth is opengl-flipped
                depth = depth[::-1, :]; rgb0 = rgb0[::-1, :, :]
            K = z[f"{c}_K"][0].astype(np.float64)
            E_c2w = z[f"{c}_E_cam2world"][0].astype(np.float64)
            E_w2c = np.linalg.inv(E_c2w)
            pts, cols, _ = unproject_depth_to_points(depth, K, E_w2c, rgb=rgb0, max_points=per_cam_budget)
            # world -> base frame
            pts = (base_inv[:3, :3] @ pts.T).T + base_inv[:3, 3]
            keep = (np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 1]) < 1.2) & (pts[:, 2] > -0.05) & (pts[:, 2] < 1.2)
            cloud_pts.append(pts[keep]); cloud_cols.append(cols[keep])
            # PW camera payload — extrinsic must map BASE-frame points to cam: E_base2cam = E_w2c @ base_T
            E_base2cam = E_w2c @ base_T
            rgb_p = _cv2.resize(rgb0, (_PW_HW[1], _PW_HW[0]), interpolation=_cv2.INTER_AREA)
            dep_p = _cv2.resize(depth, (_PW_HW[1], _PW_HW[0]), interpolation=_cv2.INTER_NEAREST)
            Kp = K.copy(); Kp[0, :] *= (_PW_HW[1] / depth.shape[1]); Kp[1, :] *= (_PW_HW[0] / depth.shape[0])
            per_cam.append({"rgb": rgb_p, "depth": dep_p.astype(np.float32),
                            "K": Kp.astype(np.float64), "E": E_base2cam.astype(np.float64)})
        scene0 = np.concatenate(cloud_pts, 0)                       # merged cloud in BASE frame
        scene_colors0 = np.concatenate(cloud_cols, 0)
        Ns = scene0.shape[0]
        T = horizon
        scene_flows = np.tile(scene0[None], (T, 1, 1)).astype(np.float32)
        scene_exists = np.ones((T, Ns), bool)
        scene_colors = np.tile((np.clip(scene_colors0, 0, 1))[None], (T, 1, 1)).astype(np.float32)
        scene_normals = np.zeros((T, Ns, 3), np.float32)

        # robot cloud: FK points are ALREADY in the base frame — keep them (no base_T transform),
        # so they align with the official URDF mesh (also FK'd from base origin).
        ti = z["ti"]
        js = np.asarray(joint_states)[ti]; gs = np.asarray(gripper_states)[ti]
        rf_out = self._robot_flows_full(js, gs, T)
        robot_flows = rf_out["robot_flows"].astype(np.float32)     # [T,Nr,3] base frame
        robot_colors, robot_normals = rf_out["robot_colors"], rf_out["robot_normals"]
        Nr = robot_flows.shape[1]
        robot_exists = np.ones((T, Nr), bool)
        gopen = np.asarray(gs, np.float64).mean(axis=1).reshape(-1, 1).astype(np.float32)

        feat_sample = {
            "scene_flows": scene_flows, "scene_colors": scene_colors, "scene_normals": scene_normals,
            "robot_flows": robot_flows, "robot_colors": robot_colors, "robot_normals": robot_normals,
            "right_gripper_open": gopen,
        }
        gather_features(feat_sample, domain=self.domain)
        sf = feat_sample["scene_features"]
        scene_features = np.tile(sf, (T, 1, 1)) if sf.shape[0] == 1 else sf

        data_dict = {
            "scene_flows": scene_flows.astype(np.float32),
            "scene_features": scene_features.astype(np.float32),
            "scene_exists": scene_exists,
            "scene_supervised_mask": np.ones((T, Ns), dtype=bool),
            "robot_flows": robot_flows,
            "robot_features": feat_sample["robot_features"].astype(np.float32),
            "robot_exists": robot_exists,
            "__domain__": self.domain,
            "_scene_colors_u8": (np.clip(scene_colors0, 0, 1) * 255).astype(np.uint8),
        }
        # emit each view as cam0, cam1, ... for PW's multi-view scene encoder
        for ci, pc in enumerate(per_cam):
            data_dict[f"cam{ci}_initial_rgb"] = pc["rgb"]
            data_dict[f"cam{ci}_initial_depth"] = pc["depth"]
            data_dict[f"cam{ci}_intrinsic"] = pc["K"]
            data_dict[f"cam{ci}_extrinsic"] = pc["E"]
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
