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


# [LangPointWorld] LIBERO gripper → [0,1] OPEN FRACTION (matches droid's gripper_open semantics,
# droid mean ~0.36). BUG (audit 2026-07-03): LIBERO's two jaws are symmetric (jaw0∈[+0.001,+0.04],
# jaw1∈[-0.04,-0.001]) so mean(jaws) CANCELS to ~0 → gripper_open feature was dead. Use total
# opening |j0|+|j1| robust-normalized by the libero_object p1/p99 range. MUST be used identically in
# the packer (pw_make_finetune_hdf5) AND build_from_sim (train/inference consistency).
_GRIP_OPEN_MIN, _GRIP_OPEN_MAX = 0.0142, 0.0798   # libero_object p1/p99 of |j0|+|j1|


def libero_gripper_open(gripper_states):
    """gripper_states[T,2] → open fraction [0,1] per frame (T,)."""
    gs = np.asarray(gripper_states, np.float64)
    opening = np.abs(gs[:, 0]) + np.abs(gs[:, 1])
    return np.clip((opening - _GRIP_OPEN_MIN) / (_GRIP_OPEN_MAX - _GRIP_OPEN_MIN), 0.0, 1.0).astype(np.float32)


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

    # LIBERO's actual robot is a Panda arm + PANDA gripper (not PW's Robotiq 2F-85). Using PW's
    # Robotiq URDF for robot_flows FK put the EEF ~4.7cm off. This fixed-mesh-path Panda URDF makes
    # robot_flows track the real LIBERO ee_pos to <1cm. See [[pw-robot-urdf-eef-mismatch]].
    _PANDA_URDF = os.path.join(os.path.dirname(__file__), "assets", "panda_arm_hand_fixed.urdf")
    # panda_link0 (URDF FK origin) → sim robot0_base origin offset, SOLVED from LIBERO ground truth:
    # matched the gripper_only Panda FK gripper-cloud centroid against the REAL gripper seg-point
    # centroid (from sim per-pixel segmentation) over all frames — offset ≈0, std<1.5cm = the FK
    # gripper cloud already aligns with the real Panda gripper to <1cm. This is the EEF-accurate fix.
    _panda_link0_to_simbase = np.array([-0.007, 0.0, -0.009], np.float32)

    def __init__(self, domain="droid", device="cuda", max_scene_points=8192, max_robot_points=512,
                 robot_urdf="panda"):
        _ensure_pw_on_path()
        self._prev_cwd = os.getcwd()
        os.chdir(_PW_ROOT)  # droid URDF path is relative to repo root
        from robot_sampler import RobotSampler
        from utils import resolve_robot_urdf
        self.domain = domain
        self.device = device
        self.max_scene_points = max_scene_points
        self.max_robot_points = max_robot_points
        # robot_urdf: "panda" = LIBERO's real Panda gripper (EEF-accurate); "robotiq"/"droid" = PW default.
        self.robot_urdf_kind = robot_urdf
        if robot_urdf == "panda":
            # gripper_only=True to MATCH official PW (RELEASE_GRIPPER_ONLY=True): robot_flows is the
            # GRIPPER point cloud, not the full arm.
            self.robot_sampler = RobotSampler(self._PANDA_URDF, gripper_only=True, device=device)
        else:
            self.robot_sampler = RobotSampler(resolve_robot_urdf(domain), device=device)

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
        gopen = libero_gripper_open(gripper_states).reshape(-1, 1)   # [0,1] open fraction (audit fix)
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
        gopen = libero_gripper_open(gs).reshape(-1, 1)   # [0,1] open fraction (audit fix)

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
                       cams=("agentview", "robot0_eye_in_hand"), flip_depth=True, frame="base",
                       sampling="uniform", importance_split=(0.45, 0.15, 0.15, 0.25),
                       official_preprocess=False, grid_size=0.015, official_max_points=12000,
                       official_importance=False):
        """Build a data_dict from tools/pw_sim_render.py output. Scene cloud = MULTI-VIEW: each
        camera's sim REAL metric depth unprojected with its sim REAL K/E (cam->world) and MERGED.
        Robot cloud = FK points. `cams` = the PW views (cam0, cam1, ...). Single-view = one cam.

        `frame`: "base" (default) expresses scene+robot in ROBOT-BASE frame (matches the official
        viz URDF mesh, good for visualization). "world" expresses scene in WORLD frame and robot in
        world too — this MUST be used for the finetuned model, because the finetune-data packer
        (pw_make_finetune_hdf5.py) stores scene_flows in WORLD frame; feeding base-frame points to
        the world-trained model is a ~0.6m OOD shift that collapses predictions to ~0 (S0-e bug)."""
        import cv2 as _cv2
        _ensure_pw_on_path()
        from dataset_components.robot import gather_features
        _PW_HW = (180, 320)

        z = sim_npz
        names = list(z["obj_names"])
        base_T = z["obj_poses"][0][names.index("robot0_base")]      # robot base -> world (t=0)
        base_inv = np.linalg.inv(base_T)                            # world -> base
        to_scene = base_inv if frame == "base" else np.eye(4)       # world->base OR world (identity)

        # Everything is expressed in the ROBOT-BASE frame (base at origin), matching the official
        # viz's URDF mesh which is FK'd from the base origin. Scene: world -> base. Robot: keep FK
        # points (already base frame). Camera extrinsic: base -> cam = (world->cam) @ (base->world).
        cams = tuple(cams) if isinstance(cams, (list, tuple)) else (cams,)
        per_cam = []            # payloads for PW cam0/cam1/...
        cloud_pts, cloud_cols, cloud_roles = [], [], []
        per_cam_budget = max(1, self.max_scene_points // len(cams))
        # ALWAYS use per-pixel seg to assign roles and DROP ROBOT points — the scene cloud must NOT
        # contain the arm/gripper (those live in robot_flows; keeping them double-counts the robot and
        # mismatches training, where the packer did `roles != "robot"`). Roles: -1=robot(dropped),
        # 0=task-obj, 1=gripper-adjacent-object(none here), 2=other-obj, 3=bg. (Bug found 2026-07-03:
        # the old uniform path kept ~11% robot points in scene_flows.)
        name2idx = {n: i for i, n in enumerate(names)}
        import json as _json
        _b2n = _json.loads(str(z["bodyid_to_name"])); _name_by_bid = {int(k): v for k, v in _b2n.items()}
        op_all = z["obj_poses"]
        _cand = [n for n in name2idx if not n.startswith(("robot0", "gripper", "mount"))
                 and n not in ("floor", "world", "table")]
        _disp = {n: np.linalg.norm(op_all[-1][name2idx[n]][:3, 3] - op_all[0][name2idx[n]][:3, 3]) for n in _cand}
        _task_obj = max(_disp, key=_disp.get) if _cand else None

        def _role(bid):
            nm = _name_by_bid.get(int(bid), "")
            if nm.startswith(("robot0", "gripper", "mount")):
                return -1                                          # ROBOT -> dropped from scene
            if nm == _task_obj: return 0
            if nm in name2idx and nm not in ("floor", "world", "table"): return 2
            return 3
        for c in cams:
            depth = z[f"{c}_depth"][0].copy()
            rgb0 = z[f"{c}_rgb"][0]
            if flip_depth:                                         # robosuite depth is opengl-flipped
                depth = depth[::-1, :]; rgb0 = rgb0[::-1, :, :]
            K = z[f"{c}_K"][0].astype(np.float64)
            E_c2w = z[f"{c}_E_cam2world"][0].astype(np.float64)
            E_w2c = np.linalg.inv(E_c2w)
            # unproject ALL valid pixels; assign role via seg (seg is NOT opengl-flipped)
            seg = z[f"{c}_seg_bodyid"][0]
            ys, xs = np.nonzero(np.isfinite(depth) & (depth > 1e-4) & (depth < 5))
            zz = depth[ys, xs]; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pcam = np.stack([(xs - cx) / fx * zz, (ys - cy) / fy * zz, zz], -1)
            ptsw = (E_c2w @ np.concatenate([pcam, np.ones((len(pcam), 1))], -1).T).T[:, :3]
            cols = rgb0[ys, xs].astype(np.float32) / 255.0
            roles = np.array([_role(b) for b in seg[ys, xs]], np.int8)
            pts = (to_scene[:3, :3] @ ptsw.T).T + to_scene[:3, 3]
            # DROP robot points (role -1) from the scene cloud
            not_robot = roles != -1
            pts, cols, roles = pts[not_robot], cols[not_robot], roles[not_robot]
            if frame == "base":
                keep = (np.abs(pts[:, 0]) < 1.5) & (np.abs(pts[:, 1]) < 1.2) & (pts[:, 2] > -0.05) & (pts[:, 2] < 1.2)
            else:  # world-frame table bounds (base is at world ~(-0.6,0,0))
                keep = (pts[:, 0] > -1.5) & (pts[:, 0] < 1.2) & (np.abs(pts[:, 1]) < 1.2) & (pts[:, 2] > -0.05) & (pts[:, 2] < 1.5)
            pts, cols, roles = pts[keep], cols[keep], roles[keep]
            if sampling == "importance":
                idx = self._importance_indices(roles, per_cam_budget, importance_split)
                pts, cols = pts[idx], cols[idx]
            elif len(pts) > per_cam_budget:                        # uniform random subsample to budget
                rng = np.random.default_rng(0)
                sel = rng.choice(len(pts), per_cam_budget, replace=False)
                pts, cols, roles = pts[sel], cols[sel], roles[sel]
            cloud_pts.append(pts); cloud_cols.append(cols); cloud_roles.append(roles)
            # PW camera payload — extrinsic must map SCENE-frame points to cam: E_scene2cam = E_w2c @ scene2world
            scene2world = base_T if frame == "base" else np.eye(4)
            E_scene2cam = E_w2c @ scene2world
            rgb_p = _cv2.resize(rgb0, (_PW_HW[1], _PW_HW[0]), interpolation=_cv2.INTER_AREA)
            dep_p = _cv2.resize(depth, (_PW_HW[1], _PW_HW[0]), interpolation=_cv2.INTER_NEAREST)
            Kp = K.copy(); Kp[0, :] *= (_PW_HW[1] / depth.shape[1]); Kp[1, :] *= (_PW_HW[0] / depth.shape[0])
            per_cam.append({"rgb": rgb_p, "depth": dep_p.astype(np.float32),
                            "K": Kp.astype(np.float64), "E": E_scene2cam.astype(np.float64)})
        scene0 = np.concatenate(cloud_pts, 0)                       # merged cloud in BASE frame
        scene_colors0 = np.concatenate(cloud_cols, 0)
        scene_roles0 = np.concatenate(cloud_roles, 0)               # per-scene-point role (0=task-obj..3=bg)
        Ns = scene0.shape[0]
        T = horizon
        scene_flows = np.tile(scene0[None], (T, 1, 1)).astype(np.float32)
        scene_exists = np.ones((T, Ns), bool)
        scene_colors = np.tile((np.clip(scene_colors0, 0, 1))[None], (T, 1, 1)).astype(np.float32)
        scene_normals = np.zeros((T, Ns, 3), np.float32)

        # robot cloud: FK points are in the base frame. Keep base for frame="base" (aligns with the
        # official URDF mesh FK'd from base origin); transform base->world for frame="world" so
        # robot and scene share the world frame the finetuned model expects.
        ti = z["ti"]
        js = np.asarray(joint_states)[ti]; gs = np.asarray(gripper_states)[ti]
        rf_out = self._robot_flows_full(js, gs, T)
        robot_flows = rf_out["robot_flows"].astype(np.float32)     # [T,Nr,3] base frame
        if frame == "world":
            robot_flows = (base_T[:3, :3] @ robot_flows.reshape(-1, 3).T).T.reshape(robot_flows.shape) + base_T[:3, 3]
            robot_flows = robot_flows.astype(np.float32)
        robot_colors, robot_normals = rf_out["robot_colors"], rf_out["robot_normals"]
        Nr = robot_flows.shape[1]
        robot_exists = np.ones((T, Nr), bool)
        gopen = libero_gripper_open(gs).reshape(-1, 1)   # [0,1] open fraction (audit fix)

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
            "scene_colors": scene_colors.astype(np.float32),          # [T,Ns,3] in [0,1] (for official xforms)
            "scene_normals": scene_normals.astype(np.float32),
            "scene_exists": scene_exists,
            "scene_supervised_mask": np.ones((T, Ns), dtype=bool),
            "robot_flows": robot_flows,
            "robot_features": feat_sample["robot_features"].astype(np.float32),
            "robot_colors": robot_colors.astype(np.float32),
            "robot_normals": robot_normals.astype(np.float32),
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

        if official_preprocess:
            self._gopen_cache = gopen                              # for feature recompute after downsample
            self._scene_roles_cache = scene_roles0                 # for post-voxel importance reallocation
            data_dict = self._apply_official_preprocess(data_dict, grid_size, official_max_points,
                                                        importance=official_importance,
                                                        importance_split=importance_split)
        return data_dict

    def _apply_official_preprocess(self, data_dict, grid_size, max_points,
                                   importance=False, importance_split=(0.45, 0.15, 0.15, 0.25)):
        """Run the OFFICIAL PointWorld test-mode scene-point preprocessing (dataset_components/
        transforms.py), so the cloud matches what PW was trained on: center_shift (mean-center
        scene+robot, adjust extrinsics) -> voxel grid_sample (1.5cm, one pt/voxel, t=0 idx applied to
        all frames) -> enforce_max_num_points (<=12000) -> center_shift -> normalize_colors.
        `importance`: AFTER voxel (which uniformizes density), do a final role-weighted subsample so
        the task object keeps ALL its voxels and background is thinned — combines official density
        control with foveation. We carry per-point roles through as a scene_ field so the official
        transforms subsample it alongside scene_flows."""
        _ensure_pw_on_path()
        from dataset_components.transforms import (
            center_shift, grid_sample_transform, enforce_max_num_points, normalize_colors)
        from dataset_components.robot import gather_features
        s = dict(data_dict)
        s.setdefault("__key__", "libero_inference")
        # BUG FIX: official normalize_colors does `colors/255.0`, so it expects [0,255] input. Our
        # scene/robot colors are [0,1] → convert to [0,255] BEFORE the official transforms so the
        # final /255 lands back in [0,1] (else colors collapse to ~0 and scene_features lose color).
        s["scene_colors"] = (np.clip(s["scene_colors"], 0, 1) * 255.0).astype(np.float32)
        s["robot_colors"] = (np.clip(s["robot_colors"], 0, 1) * 255.0).astype(np.float32)
        # remember original t0 scene points + roles to recover per-point role AFTER voxel (grid_sample
        # only carries a hardcoded set of scene_ fields, NOT arbitrary ones — so match by coordinate).
        orig_pts0 = s["scene_flows"][0].copy()
        orig_roles = self._scene_roles_cache.astype(np.int64)
        s = center_shift(s)
        if importance:
            # FOVEATED voxel: split scene by role, keep ALL task-object points at FULL resolution (skip
            # voxel), voxel-downsample only the rest. Then this is the dense-object cloud PW sees. This
            # must happen BEFORE voxel-on-everything (which would already collapse the sparse obj points).
            obj_mask = orig_roles == 0
            s = self._foveated_voxel(s, obj_mask, grid_size, grid_sample_transform)
        else:
            s = grid_sample_transform(s, grid_size=grid_size, mode="test")   # voxels BOTH scene AND robot
        s = enforce_max_num_points(s, max_scene_points=max_points, deterministic=True, seed=0)
        # NOTE: do NOT restore/keep a full-res robot cloud here. The official pipeline voxel-downsamples
        # the robot alongside the scene (grid_sample handles both prefixes) so the SECOND center_shift
        # re-centers scene+robot CONSISTENTLY. Keeping a full-res robot (earlier bug) desynced it from
        # the voxel-shifted scene centroid → robot ended up ~0.3m off. Let robot follow official flow.
        s = center_shift(s)
        s = normalize_colors(s)
        # scene_features/robot_features were computed on the FULL cloud (pre-downsample) → stale/mismatched
        # shapes. Official order gathers features AFTER density control, so recompute them here on the
        # now-downsampled clouds (fixes IndexError: features Ns != scene_flows Ns).
        Tn = s["scene_flows"].shape[0]
        feat = {
            "scene_flows": s["scene_flows"], "scene_colors": s["scene_colors"], "scene_normals": s["scene_normals"],
            "robot_flows": s["robot_flows"], "robot_colors": s["robot_colors"], "robot_normals": s["robot_normals"],
            "right_gripper_open": self._gopen_cache,
        }
        gather_features(feat, domain=self.domain)
        sf = feat["scene_features"]
        s["scene_features"] = (np.tile(sf, (Tn, 1, 1)) if sf.shape[0] == 1 else sf).astype(np.float32)
        s["robot_features"] = feat["robot_features"].astype(np.float32)
        # exists masks must match new Ns/Nr
        s["scene_exists"] = np.ones(s["scene_flows"].shape[:2], bool)
        s["scene_supervised_mask"] = np.ones(s["scene_flows"].shape[:2], bool)
        s["robot_exists"] = np.ones(s["robot_flows"].shape[:2], bool)
        # keep the u8 color mirror in sync with the (possibly downsampled) scene for viz
        if "scene_colors" in s:
            sc = s["scene_colors"]; col0 = sc[0] if sc.ndim == 3 else sc
            s["_scene_colors_u8"] = (np.clip(col0, 0, 1) * 255).astype(np.uint8)
        return s

    @staticmethod
    def _foveated_voxel(s, obj_mask, grid_size, grid_sample_transform):
        """Keep ALL task-object scene points at full resolution; voxel-downsample the rest; concat.
        Operates on the scene_ per-point fields (flows/colors/normals + exists/masks)."""
        scene_keys = [k for k in s if k.startswith("scene_") and isinstance(s[k], np.ndarray)
                      and s[k].ndim >= 2 and s[k].shape[1] == obj_mask.shape[0]]
        obj_part = {k: s[k][:, obj_mask] for k in scene_keys}       # object: untouched (full res)
        rest = dict(s)
        for k in scene_keys:
            rest[k] = s[k][:, ~obj_mask]
        rest = grid_sample_transform(rest, grid_size=grid_size, mode="test")  # voxel only the non-object
        for k in scene_keys:
            s[k] = np.concatenate([obj_part[k], rest[k]], axis=1)   # object first, then voxelized rest
        return s

    @staticmethod
    def _importance_indices(roles, budget, split):
        """Allocate `budget` point indices across roles (0=task-obj,1=gripper,2=other-obj,3=bg) by
        `split`, keeping ALL task-obj points, topping up from background if short. Same total budget
        as uniform (in-distribution for PW) but reallocated dense-on-object."""
        rng = np.random.default_rng(0)
        alloc = [int(budget * s) for s in split]
        out = []
        for r, k in enumerate(alloc):
            pool = np.where(roles == r)[0]
            if len(pool) == 0:
                continue
            take = len(pool) if (r == 0 or len(pool) <= k) else k   # keep ALL task-obj points
            out.append(pool if take >= len(pool) else rng.choice(pool, take, replace=False))
        idx = np.concatenate(out) if out else np.arange(min(budget, len(roles)))
        if len(idx) < budget:                                       # top up from remaining (mostly bg)
            rest = np.setdiff1d(np.arange(len(roles)), idx)
            if len(rest):
                idx = np.concatenate([idx, rng.choice(rest, min(budget - len(idx), len(rest)), replace=False)])
        return idx

    def _robot_flows_full(self, joint_states, gripper_states, T):
        """Like build_robot_flows but returns colors/normals too, padded to T frames.
        robot_urdf=="panda": FK with LIBERO's real Panda gripper (EEF-accurate, <1cm vs ee_pos);
        else: PW's Robotiq via _get_robot_flows_droid (legacy, ~4.7cm EEF error)."""
        _ensure_pw_on_path()
        def _pad(a):
            return a if a.shape[0] >= T else np.concatenate([a, np.repeat(a[-1:], T - a.shape[0], 0)], 0)
        js = np.asarray(joint_states, np.float64)
        gs = np.asarray(gripper_states, np.float64)
        if self.robot_urdf_kind == "panda":
            import torch as _t
            rs = self.robot_sampler
            rs.presample(self.max_robot_points, seed=0)
            J = _t.tensor(js[:, :7], dtype=_t.float32, device=self.device)     # (B,7) radians
            G = _t.tensor(gs, dtype=_t.float32, device=self.device)            # (B,2)
            jd = {f"panda_joint{i+1}": J[:, i].contiguous() for i in range(7)}
            jd["panda_finger_joint1"] = G[:, 0].abs().contiguous()
            jd["panda_finger_joint2"] = G[:, 1].abs().contiguous()
            assert set(rs.joint_names) <= set(jd), \
                f"joint_dict keys must cover rs.joint_names (silent-zero-pose bug guard); missing {set(rs.joint_names)-set(jd)}"
            pts, cols, norms = rs.compute_points(jd)                           # (B,N,3) panda_link0 frame
            rf = pts.detach().cpu().numpy().astype(np.float32)
            # panda_link0 origin sits ~7.2cm above the sim robot0_base origin (pedestal). Shift FK
            # points into the sim base frame the scene uses (build_from_sim then applies base_T→world).
            rf = rf + self._panda_link0_to_simbase
            rc = cols.detach().cpu().numpy().astype(np.float32) / 255.0 if cols is not None else np.zeros_like(rf)
            rn = norms.detach().cpu().numpy().astype(np.float32) if norms is not None else np.zeros_like(rf)
            return {"robot_flows": _pad(rf), "robot_colors": _pad(rc), "robot_normals": _pad(rn)}
        from dataset_components.robot import _get_robot_flows_droid
        sample = {"joint_positions": js, "gripper_positions": gs.mean(axis=1),
                  "joint_names": [f"panda_joint{i}" for i in range(1, 8)]}
        out = _get_robot_flows_droid(sample, self.robot_sampler, self.max_robot_points)
        return {"robot_flows": _pad(np.asarray(out["robot_flows"], np.float32)),
                "robot_colors": _pad(np.asarray(out["robot_colors"], np.float32)),
                "robot_normals": _pad(np.asarray(out["robot_normals"], np.float32))}

    def __del__(self):
        try:
            os.chdir(self._prev_cwd)
        except Exception:
            pass
