# starVLA/model/framework/VLM4A/GeoMemoryVLA.py
# [Geo-MemoryVLA] Framework orchestrator: frozen VGGT world-state + Qwen3-VL
# semantic stream + dual memory + 3D imagination, fused into the [B,L,H] condition
# consumed by the unchanged GR00T flow-matching head. Skeleton from QwenGR00T.
# Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.modules.geomem.condition_assembler import ConditionAssembler
from starVLA.model.modules.geomem.imagination_adapter import ImaginationAdapter
from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
from starVLA.model.modules.memory.dual_memory_bank import DualMemoryBank
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class GeoMemoryVLADefaultConfig:
    name: str = "GeoMemoryVLA"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
        "vl_hidden_dim": 2048,
    })
    world_state: dict = field(default_factory=lambda: {
        "enabled": True, "stream": "dual",
        "model_name": "facebook/VGGT-1B", "layer_index": 4, "num_cameras": 2,
    })
    memory: dict = field(default_factory=lambda: {
        "enabled": True, "mem_length": 16, "retrieval_layers": 2,
        "fusion_type": "gate", "consolidate_type": "tome",
    })
    imagination: dict = field(default_factory=lambda: {
        "enabled": True, "horizon": 4, "depth": 4, "steps": 4, "loss_scale": 0.5,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B", "action_hidden_dim": 1024, "hidden_size": 1024,
        "add_pos_embed": True, "max_seq_len": 2048, "action_dim": 7, "state_dim": 7,
        "action_horizon": 8, "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
        "num_timestep_buckets": 1000, "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            "cross_attention_dim": 1024, "dropout": 0.2, "final_dropout": True,
            "interleave_self_attention": True, "norm_type": "ada_norm",
            "num_layers": 16, "output_dim": 1024, "positional_embeddings": None,
        },
    })


@FRAMEWORK_REGISTRY.register("GeoMemoryVLA")
class GeoMemoryVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(GeoMemoryVLADefaultConfig, config)
        fw = self.config.framework

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        sem_dim = int(self.qwen_vl_interface.model.config.hidden_size)

        self.use_geo = fw.world_state["enabled"] and fw.world_state["stream"] in ("geo_only", "dual")
        self.use_sem = fw.world_state["stream"] in ("sem_only", "dual")
        if self.use_geo:
            # Vendored frozen VGGT-World backbone (Task 1 adapter).
            self.world_state = WorldStateAdapter(pretrained_vggt_repo=fw.world_state["model_name"])
            geo_dim = self.world_state.hidden_size  # 1024
        else:
            geo_dim = sem_dim

        self.use_memory = fw.memory["enabled"]
        if self.use_memory:
            self.memory = DualMemoryBank(
                geo_dim=geo_dim, sem_dim=sem_dim,
                mem_length=fw.memory["mem_length"],
                retrieval_layers=fw.memory["retrieval_layers"],
                fusion_type=fw.memory["fusion_type"],
                consolidate_type=fw.memory["consolidate_type"],
            )

        self.use_imag = fw.imagination["enabled"] and self.use_geo
        if self.use_imag:
            # Vendored VGGTWorldModel wrapper (Task 5 adapter). chunk/context drive the
            # imagined horizon; it consumes multi-frame image windows (see Task 5 note).
            self.imaginer = ImaginationAdapter(
                pretrained_vggt_repo=fw.world_state["model_name"],
                chunk_size=int(fw.imagination["horizon"]),
                context_size=int(fw.imagination.get("context_size", 2)),
            )
        self.imag_loss_scale = float(fw.imagination["loss_scale"])
        self.imag_horizon = int(fw.imagination["horizon"])

        # Condition hidden = GR00T cross_attention_dim. Build assembler over active streams.
        cond_dim = int(fw.action_model.diffusion_model_cfg.cross_attention_dim)
        stream_dims = {}
        if self.use_geo:
            stream_dims["geo"] = geo_dim
            if self.use_memory:
                stream_dims["m_geo"] = geo_dim
            if self.use_imag:
                stream_dims["imag"] = geo_dim
        if self.use_sem:
            stream_dims["sem"] = sem_dim
            if self.use_memory:
                stream_dims["m_sem"] = sem_dim
        self.assembler = ConditionAssembler(stream_dims=stream_dims, out_dim=cond_dim)

        self.action_model = get_action_model(config=self.config)
        self.action_horizon = int(fw.action_model.action_horizon)

    # --- helpers -------------------------------------------------------------
    def _encode(self, examples):
        images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        sem = out.hidden_states[-1] if self.use_sem else None  # [B, L, H]
        geo = None
        if self.use_geo:
            pix = self._build_vggt_pixels(images, device=out.hidden_states[-1].device)
            state = self.world_state.encode(pix)            # GeometryState (frozen)
            geo = self.world_state.flatten(state)            # [B, F*2*tokens, 1024]
        return geo, sem

    def _build_vggt_pixels(self, images, device):
        # images: List[B][List[PIL]] -> [B, num_views, 3, H, W] in [0,1].
        import torchvision.transforms.functional as TF
        batch = []
        for views in images:
            vs = [TF.to_tensor(v.convert("RGB")) for v in views]
            batch.append(torch.stack(vs, dim=0))
        return torch.stack(batch, dim=0).to(device)

    def _build_image_window(self, examples, device):
        # Multi-frame window for the world model. Prefer an explicit "image_window"
        # (List[frames][views] of PIL) if the dataloader supplies it; else degenerate
        # to the current single frame's views (Task 5 note / Phase-C follow-up).
        import torchvision.transforms.functional as TF
        windows = []
        for e in examples:
            frames = e.get("image_window", [e["image"]])
            frame_tensors = []
            for views in frames:
                vs = [TF.to_tensor(v.convert("RGB")) for v in views]
                frame_tensors.append(torch.stack(vs, dim=0))   # [num_views, 3, H, W]
            # Flatten (frames, views) into the world model's frame axis.
            windows.append(torch.cat(frame_tensors, dim=0))
        return torch.stack(windows, dim=0).to(device)          # [B, F*views, 3, H, W]

    def _assemble(self, geo, sem, m_geo, m_sem, imag):
        streams = {}
        if self.use_geo:
            streams["geo"] = geo
            if self.use_memory:
                streams["m_geo"] = m_geo
            if self.use_imag:
                streams["imag"] = imag
        if self.use_sem:
            streams["sem"] = sem
            if self.use_memory:
                streams["m_sem"] = m_sem
        return self.assembler(streams)

    # --- training ------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs):
        geo, sem = self._encode(examples)
        episode_ids = [e.get("episode_id", 0) for e in examples]
        timesteps = [e.get("timestep", 0) for e in examples]

        m_geo = m_sem = None
        if self.use_memory:
            m_geo, m_sem = self.memory.process(geo, sem, episode_ids, timesteps)

        device = (sem if sem is not None else geo).device
        imag = None
        imag_loss = torch.zeros((), device=device)
        if self.use_imag:
            # The vendored world model consumes a multi-frame image window. When the
            # dataloader provides one ("image_window" key: context+chunk frames), pass it;
            # otherwise fall back to the current views (degenerate single-step, Task 5 note).
            # `where` drives the stage-1/stage-2 flow-forcing curriculum.
            window = self._build_image_window(examples, device=device)
            imag = self.imaginer.imagine_tokens(window, forecast_frames=self.imag_horizon)
            imag_loss = self.imaginer.training_loss(window, where=float(kwargs.get("where", 0.0)))

        cond, mask = self._assemble(geo, sem, m_geo, m_sem, imag)

        actions = [e["action"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None
        rep = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=cond.device, dtype=cond.dtype)
            actions_target = actions[:, -self.action_horizon :, :].repeat(rep, 1, 1)
            cond_r = cond.repeat(rep, 1, 1)
            mask_r = mask.repeat(rep, 1)
            state_r = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=cond.device, dtype=cond.dtype)
                state_r = state_t.repeat(rep, 1, 1)
            action_loss = self.action_model(cond_r, actions_target, state_r, encoder_attention_mask=mask_r)

        out = {"action_loss": action_loss}
        if self.use_imag:
            out["imagination_loss"] = self.imag_loss_scale * imag_loss
        return out

    # --- inference -----------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if type(examples) is not list:
            examples = [examples]
        for e in examples:
            e["image"] = to_pil_preserve(e["image"])
        train_obs = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs:
            for e in examples:
                e["image"] = resize_images(e["image"], target_size=train_obs)

        geo, sem = self._encode(examples)
        episode_ids = [e.get("episode_id", 0) for e in examples]
        timesteps = [e.get("timestep", 0) for e in examples]
        m_geo = m_sem = None
        if self.use_memory:
            m_geo, m_sem = self.memory.process(geo, sem, episode_ids, timesteps)
        imag = None
        if self.use_imag:
            window = self._build_image_window(examples, device=geo.device)
            imag = self.imaginer.imagine_tokens(window, forecast_frames=self.imag_horizon)
        cond, mask = self._assemble(geo, sem, m_geo, m_sem, imag)

        state = [e["state"] for e in examples] if "state" in examples[0] else None
        state_t = torch.from_numpy(np.array(state)).to(cond.device, dtype=cond.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(cond, state_t, encoder_attention_mask=mask)
        return {"normalized_actions": pred.detach().cpu().numpy()}
