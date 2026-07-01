# [LangPointWorld] VGGT geometry provider — depth + camera intrinsics/extrinsics from
# LIBERO RGB (Phase-0 decision: VGGT primary depth source, in-env, GPU-native).
# Resolves both the LIBERO-has-no-depth gap AND the missing-intrinsics problem in one model.
# Vendored VGGT from /workspace/tingting/vggt-world (facebook/VGGT-1B, cached in HF_HOME).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import numpy as np
import torch
import h5py
import cv2

_VGGT_SIZE = 518  # VGGT requires input dims divisible by patch=14; 518 is its target


class VGGTGeometryProvider:
    """Runs frozen VGGT once per (episode, camera) sequence; serves per-timestep
    metric depth + 3x3 intrinsics + 4x4 extrinsics. Depth/K/E cached in-memory per key."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from vggt.models.vggt import VGGT
        self.device = device
        self.dtype = dtype
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._cache = {}  # (hdf5_path, demo, cam_key) -> {"depth":[T,H,W], "K":[T,3,3], "E":[T,4,4]}

    @staticmethod
    def _load_rgb_seq(hdf5_path, demo, cam_key):
        with h5py.File(hdf5_path, "r") as f:
            rgb = np.asarray(f["data"][demo]["obs"][cam_key][:])  # [T,H,W,3] uint8, opengl (flipped)
        rgb = rgb[:, ::-1, :, :].copy()                            # un-flip (opengl convention)
        return rgb

    @torch.no_grad()
    def _run_vggt(self, hdf5_path, demo, cam_key):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        rgb = self._load_rgb_seq(hdf5_path, demo, cam_key)         # [T,H,W,3] native (128)
        T, H, W, _ = rgb.shape
        # VGGT needs dims divisible by 14; resize to 518x518, run, then map depth back to native.
        rgb_v = np.stack([cv2.resize(f, (_VGGT_SIZE, _VGGT_SIZE), interpolation=cv2.INTER_AREA)
                          for f in rgb], 0)
        imgs = torch.from_numpy(rgb_v).float().div(255.0).permute(0, 3, 1, 2).to(self.device)  # [T,3,518,518]
        with torch.autocast("cuda", dtype=self.dtype):
            pred = self.model(imgs)  # adds batch dim internally -> [1,T,...]
        depth_v = pred["depth"][0, ..., 0].float().cpu().numpy()   # [T,518,518]
        # resize depth back to native LIBERO resolution so it aligns with the RGB the adapter reads
        depth = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST) for d in depth_v], 0)  # [T,H,W]
        # intrinsics were produced at 518; rescale focal/principal to native pixels
        ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(_VGGT_SIZE, _VGGT_SIZE))
        intr = intr.clone()
        intr[..., 0, :] *= (W / _VGGT_SIZE)   # fx, cx  -> native x scale
        intr[..., 1, :] *= (H / _VGGT_SIZE)   # fy, cy  -> native y scale
        ext = ext[0].float().cpu().numpy()                          # [T,3,4]
        intr = intr[0].float().cpu().numpy()                        # [T,3,3]
        # promote extrinsics [3,4] -> [4,4]
        E = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
        E[:, :3, :4] = ext
        return {"depth": depth, "K": intr.astype(np.float64), "E": E}

    def _ensure(self, hdf5_path, demo, cam_key):
        key = (hdf5_path, demo, cam_key)
        if key not in self._cache:
            self._cache[key] = self._run_vggt(hdf5_path, demo, cam_key)
        return self._cache[key]

    def depth_provider(self, hdf5_path, demo):
        """Returns a callable (cam_key, t) -> HxW float32 depth, matching build_pw_sample's
        depth_provider contract. VGGT camera keys map from LIBERO obs keys."""
        def _fn(cam_key, t):
            g = self._ensure(hdf5_path, demo, cam_key)
            return g["depth"][t].astype(np.float32)
        return _fn

    def intrinsics_extrinsics(self, hdf5_path, demo, t):
        """Per-timestep K/E dicts keyed by LIBERO camera names, for build_pw_sample."""
        K, E = {}, {}
        for cam in ("agentview_rgb", "eye_in_hand_rgb"):
            g = self._ensure(hdf5_path, demo, cam)
            K[cam] = g["K"][t]
            E[cam] = g["E"][t]
        return K, E
