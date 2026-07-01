# [LangPointWorld] Frozen PointWorld teacher wrapper (C2, spec §3.1). Forward-only.
# Mirrors the minimal Trainer(inference_only=True) build path from
# PointWorld/training/trainer.py (contract -> data_info_dict from ckpt shapes ->
# BaseModel -> load_state_dict), avoiding the eval-harness/dataloader/wandb scaffolding.
# Vendored driving of NVlabs/PointWorld (Apache-2.0): github.com/NVlabs/PointWorld
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


class PointWorldTeacher:
    def __init__(self, ckpt_path, ptv3_size="large", domain="droid", device="cuda"):
        _ensure_pw_on_path()
        # PointWorld's default-arg URDF path is relative to its repo root; run from there.
        self._prev_cwd = os.getcwd()
        os.chdir(_PW_ROOT)
        from arguments import parse_args
        from pointworld.base import BaseModel
        from pointworld.checkpoint_contract import (
            read_checkpoint_contract, apply_model_contract_to_args,
        )
        self.device = device
        self.domain = domain

        # Minimal args: mimic `eval.py` invocation, then let the checkpoint contract
        # overwrite arch-defining fields so init matches the trained model exactly.
        argv_backup = sys.argv
        sys.argv = ["eval.py", "--model_path", ckpt_path, "--domains", domain,
                    "--ptv3_size", ptv3_size, "--device", device, "--distributed", "false"]
        try:
            args = parse_args()
        finally:
            sys.argv = argv_backup

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_contract, _data_contract = read_checkpoint_contract(ckpt, context="teacher")
        apply_model_contract_to_args(args, model_contract, context="teacher",
                                     explicit_cli_dests=set())

        # Infer projection dims from checkpoint weight shapes (trainer.py inference_only path).
        state = ckpt["model"]
        scene_w = state["scene_feature_encoder.scene_raw_feat_proj.weight"]
        robot_w = state["robot_proj.fc1.weight"]
        data_info = {
            "scene_features_dim": int(scene_w.shape[1]),
            "robot_features_dim": int(robot_w.shape[1]),
        }

        args.device = device
        args.distributed = False
        self.args = args
        self.model = BaseModel(args, data_info, rank=0).to(device).eval()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[LangPointWorld] teacher load: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[LangPointWorld] teacher load: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def imagine_from_datadict(self, data_dict: dict) -> dict:
        """data_dict: LiberoDataDictBuilder.build() output (numpy arrays + __domain__).
        Bypasses the WDS pipeline (data-branch-only); feeds BaseModel.forward directly.
        Returns imagined future scene-flow in the world frame."""
        batch = {}
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.as_tensor(v).unsqueeze(0).to(self.device)
            elif k == "__domain__":
                batch[k] = [v]
            else:
                batch[k] = v
        out = self.model(batch, training=False)
        return {
            "scene_flows": out["scene_flows"][0].detach().cpu(),       # [T,Ns,3]
            "scene_coord0": batch["scene_flows"][0, 0].detach().cpu(),  # [Ns,3]
        }

    def __del__(self):
        try:
            os.chdir(self._prev_cwd)
        except Exception:
            pass
