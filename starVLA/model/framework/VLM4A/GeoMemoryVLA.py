# starVLA/model/framework/VLM4A/GeoMemoryVLA.py
# [Geo-MemoryVLA] Framework orchestrator: frozen VGGT world-state + Qwen3-VL
# semantic stream + dual memory + 3D imagination, fused into the [B,L,H] condition
# consumed by the unchanged GR00T flow-matching head. Skeleton from QwenGR00T.
# Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import warnings
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
# [D4RT-WorldState] factory selects vggt_world vs d4rt world-state + imaginer adapters.
from starVLA.model.modules.geomem.world_state_factory import build_world_state, build_imaginer
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
        # [Geo-MemoryVLA] horizon=2 (paper m=2, VGGT-World arXiv 2603.12655). Window =
        # context+horizon+1 = 5 frames. See image-window design spec.
        "enabled": True, "horizon": 2, "depth": 4, "steps": 4, "loss_scale": 0.5,
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


# [D4RT-WorldState] Agentview-only temporal window for the monocular D4RT stream (spec §5,
# B1). The frame axis is TIME; view 0 (agentview) only — wrist reaches the policy via the
# Qwen3-VL/GR00T path, not the geometric stream. Uses image_window when present (a real
# temporal series), else replicate-pads the current agentview frame to win_len. Range [0,1],
# resized to 256x256 (D4RT input). Module-level so it is unit-testable without the model.
def _build_d4rt_window(examples, device, win_len: int = 48):
    import torchvision.transforms.functional as TF
    windows = []
    for e in examples:
        imgs = e["image"] if isinstance(e["image"], (list, tuple)) else [e["image"]]
        agent = imgs[0].convert("RGB")                       # view 0 = agentview (B1)
        frames = e.get("image_window")
        if frames:                                           # List[F][views] -> agentview series
            series = [(f[0] if isinstance(f, (list, tuple)) else f).convert("RGB") for f in frames]
        else:
            series = [agent]
        series = (series + [series[-1]] * win_len)[:win_len]  # replicate-pad to win_len
        ten = torch.stack([TF.resize(TF.to_tensor(im), [256, 256]) for im in series], dim=0)
        windows.append(ten)
    return torch.stack(windows, dim=0).to(device)             # [B, win_len, 3, 256, 256]


@FRAMEWORK_REGISTRY.register("GeoMemoryVLA")
class GeoMemoryVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(GeoMemoryVLADefaultConfig, config)
        fw = self.config.framework

        # [Geo-MemoryVLA] Determine which streams are active BEFORE constructing Qwen3-VL,
        # so that geo_only ablations can run in environments where Qwen3VLForConditionalGeneration
        # is absent (e.g. transformers < 4.54). Qwen3-VL is only constructed when use_sem=True.
        self.use_geo = fw.world_state["enabled"] and fw.world_state["stream"] in ("geo_only", "dual")
        self.use_sem = fw.world_state["stream"] in ("sem_only", "dual")

        if self.use_sem:
            self.qwen_vl_interface = get_vlm_model(config=self.config)
            sem_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        else:
            # [Geo-MemoryVLA] geo_only ablation: no VLM. Read sem_dim from config as a
            # fallback so ConditionAssembler / DualMemoryBank dim wiring still works if
            # sem streams are later enabled without changing this branch.
            sem_dim = int(fw.qwenvl.get("vl_hidden_dim", 2048))
        if self.use_geo:
            # [D4RT-WorldState] backbone-agnostic construction (vggt_world | d4rt). The
            # adapter exposes .hidden_size at construction, so geo_dim adapts to the backbone
            # (VGGT=1024, D4RT=1280) without any other change here.
            self.world_state = build_world_state(fw)
            geo_dim = self.world_state.hidden_size
            # [D4RT-WorldState] remember which backbone so _encode routes the right input
            # builder (VGGT: multi-view pixels; D4RT: agentview temporal window).
            self._backbone = fw.world_state.get("backbone", "vggt_world")
            self._d4rt_win_len = int(fw.world_state.get("clip_frames", 48))
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
            # [D4RT-WorldState] backbone-agnostic imaginer (shares the frozen world_state).
            self.imaginer = build_imaginer(fw, self.world_state)
            backbone = fw.world_state.get("backbone", "vggt_world")
            if backbone == "d4rt":
                # [D4RT-WorldState] D4RT is monocular: the window is a clip_frames-long
                # agentview temporal series (spec §5), not VGGT's context+chunk+1.
                self._imag_window_len = int(fw.world_state.get("clip_frames", 48))
            else:
                # [Geo-MemoryVLA] Window must satisfy VGGTWorldModel Stage-2: context+chunk+1.
                ctx = int(fw.imagination.get("context_size", 2))
                chunk = int(fw.imagination["horizon"])
                self._imag_window_len = ctx + chunk + 1
                # (the dataloader derives the same length from the DataConfig; this is a guard.)
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
        # [Geo-MemoryVLA] ConditionAssembler crashes on an all-None/empty stream set
        # (torch.cat of empty). A contradictory ablation (e.g. world_state.enabled=False
        # with stream="geo_only") would yield no active streams — fail fast with a clear msg.
        if not stream_dims:
            raise ValueError(
                "GeoMemoryVLA: no active condition streams — check framework.world_state.enabled "
                "and framework.world_state.stream (need at least one of geo/sem active)."
            )
        self.assembler = ConditionAssembler(stream_dims=stream_dims, out_dim=cond_dim)

        self.action_model = get_action_model(config=self.config)
        self.action_horizon = int(fw.action_model.action_horizon)

    # --- helpers -------------------------------------------------------------
    def _encode(self, examples):
        images = [e["image"] for e in examples]
        # [Geo-MemoryVLA] geo_only ablation: skip VLM forward entirely when use_sem=False
        # (Qwen3-VL may not be installed in this env). Device is inferred from the VGGT
        # backbone parameters so the geo path still runs on the correct device.
        sem_cog = None
        if self.use_sem:
            instructions = [e["lang"] for e in examples]
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
            sem = out.hidden_states[-1]  # [B, L, H]  (L varies with instruction length!)
            vggt_device = sem.device
            # [Geo-MemoryVLA] Pool the VARIABLE-length sem into a fixed single cognitive token for
            # MEMORY (the ToMe bank averages entries -> requires fixed size; different LIBERO task
            # instructions give different L, which crashed ToMe with "size a != size b"). This is
            # MemoryVLA's original design (last valid token via attention_mask). The FULL sem still
            # goes to the condition assembler; only the memory entry is pooled.
            sem_cog = self._pool_cog_token(sem, qwen_inputs.get("attention_mask", None))
        else:
            sem = None
            # [Geo-MemoryVLA] Infer device from the VGGT backbone when VLM is absent.
            vggt_device = next(self.world_state.parameters()).device

        geo = None
        if self.use_geo:
            if getattr(self, "_backbone", "vggt_world") == "d4rt":
                # [D4RT-WorldState] agentview-only temporal window -> D4RT memory [B,N,C].
                pix = _build_d4rt_window(examples, device=vggt_device, win_len=self._d4rt_win_len)
                pix = self._cast_to_vggt_dtype(pix)         # match frozen backbone dtype
            else:
                pix = self._build_vggt_pixels(images, device=vggt_device)
            state = self.world_state.encode(pix)            # GeometryState | D4RTState (frozen)
            geo = self.world_state.flatten(state)            # [B, N, C]
        return geo, sem, sem_cog

    def _pool_cog_token(self, sem, attention_mask):
        # [Geo-MemoryVLA] [B, L, H] -> [B, 1, H]: the hidden state at the last VALID token
        # (MemoryVLA's cognitive token). Uses attention_mask so it's correct under left/right pad.
        if attention_mask is None:
            return sem[:, -1:, :]
        attention_mask = attention_mask.to(sem.device)
        last_idx = attention_mask.long().cumsum(dim=1).argmax(dim=1)  # [B] last valid position
        idx = last_idx[:, None, None].expand(-1, 1, sem.shape[-1])
        return sem.gather(1, idx)  # [B, 1, H]

    def _build_vggt_pixels(self, images, device):
        # images: List[B][List[PIL]] -> [B, num_views, 3, H, W] in [0,1].
        import torchvision.transforms.functional as TF
        batch = []
        for views in images:
            vs = [TF.to_tensor(v.convert("RGB")) for v in views]
            batch.append(torch.stack(vs, dim=0))
        out = torch.stack(batch, dim=0).to(device)
        # [Geo-MemoryVLA] cast to the frozen VGGT backbone dtype so the conv matches without an
        # ambient autocast. Training autocasts forward(); predict_action (eval) does NOT, so a
        # float32 input here crashed the first eval ("Input type (float) and bias type bf16").
        return self._cast_to_vggt_dtype(out)

    def _cast_to_vggt_dtype(self, x):
        ws = getattr(self, "world_state", None)
        if ws is not None and hasattr(ws, "dtype"):
            return x.to(ws.dtype)
        return x

    def _build_image_window(self, examples, device, allow_degenerate=False):
        # [Geo-MemoryVLA] Single-camera temporal window for the world model (S1: image_window is
        # primary-only, so the frame axis is TIME, not views). Shape -> [B, F, 3, H, W].
        # TRAINING (allow_degenerate=False): a missing "image_window" while imagination is on is a
        # hard error (silent single-frame degeneration would look like imagination trains when it
        # does not). INFERENCE (allow_degenerate=True, S2): no rolling buffer exists, so replicate
        # the current frame to the window length -> a valid static temporal window, no crash.
        import torchvision.transforms.functional as TF
        win_len = int(getattr(self, "_imag_window_len", 5))
        windows = []
        for e in examples:
            if "image_window" in e:
                frames = e["image_window"]                      # List[F][views]
            elif allow_degenerate:
                # Replicate current frame's primary view to the temporal window length.
                primary = [e["image"][0]] if isinstance(e["image"], (list, tuple)) else [e["image"]]
                frames = [primary for _ in range(win_len)]
            elif getattr(self, "use_imag", False):
                raise ValueError(
                    "GeoMemoryVLA: imagination enabled but batch has no 'image_window'. "
                    "Enable the image_window dataloader modality "
                    "(datasets.vla_data.enable_image_window=true) — see "
                    "docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md"
                )
            else:
                frames = [e["image"]]
            frame_tensors = []
            for views in frames:
                vs = [TF.to_tensor(v.convert("RGB")) for v in views]
                frame_tensors.append(torch.stack(vs, dim=0))   # [num_views(=1), 3, H, W]
            windows.append(torch.cat(frame_tensors, dim=0))
        out = torch.stack(windows, dim=0).to(device)           # [B, F, 3, H, W] (single-view)
        # [Geo-MemoryVLA] S1 live guard (was dead): frame axis must equal the temporal window
        # length, i.e. frames == timesteps. Catches any view/time-axis regression.
        if getattr(self, "use_imag", False):
            assert out.shape[1] == win_len, (
                f"image_window frame count {out.shape[1]} != expected {win_len} "
                f"(single-view temporal window). See pipeline-fixes plan S1."
            )
        # [Geo-MemoryVLA] match frozen VGGT dtype (eval has no ambient autocast — see dtype fix).
        return self._cast_to_vggt_dtype(out)

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
        geo, sem, sem_cog = self._encode(examples)
        episode_ids = [e.get("episode_id", 0) for e in examples]
        timesteps = [e.get("timestep", 0) for e in examples]

        m_geo = m_sem = None
        if self.use_memory:
            # [Geo-MemoryVLA] memory stores the POOLED cog token (fixed size), not the full
            # variable-length sem — see _pool_cog_token / the ToMe varlen crash fix.
            m_geo, m_sem = self.memory.process(geo, sem_cog, episode_ids, timesteps)

        device = (sem if sem is not None else geo).device
        imag = None
        imag_loss = torch.zeros((), device=device)
        if self.use_imag:
            # [Geo-MemoryVLA] The vendored world model needs a multi-frame window. The
            # dataloader must supply "image_window" (context+chunk+1 frames) when imagination
            # is on; _build_image_window raises ValueError if it is missing (no silent
            # single-frame degeneration). `where` drives stage-1/stage-2 flow-forcing.
            # [D4RT-WorldState] D4RT uses an agentview temporal window (monocular, B1); VGGT
            # uses the context+chunk+1 multi-view window. Both imaginers accept a raw window.
            if getattr(self, "_backbone", "vggt_world") == "d4rt":
                window = _build_d4rt_window(examples, device=device, win_len=self._d4rt_win_len)
                window = self._cast_to_vggt_dtype(window)
            else:
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

        # [Geo-MemoryVLA] EVAL-AUTOCAST: training wraps the whole forward in autocast(bf16) via the
        # trainer; predict_action (@inference_mode) had none, so memory cross-attn / VGGT / imagination
        # ran in float32 against bf16 weights and crashed ("mat1/mat2 must have the same dtype"). Wrap
        # the compute body in the same bf16 autocast so every submodule matches, eliminating the whole
        # class of eval dtype mismatches (root-cause fix, not per-module casting).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            geo, sem, sem_cog = self._encode(examples)
            episode_ids = [e.get("episode_id", 0) for e in examples]
            timesteps = [e.get("timestep", 0) for e in examples]
            m_geo = m_sem = None
            if self.use_memory:
                # [Geo-MemoryVLA] B4: clear stale cross-rollout memory at an episode boundary so eval
                # rollouts don't bleed into each other. Trigger on an explicit reset kwarg or the first
                # step of a new episode (timestep == 0). Without this, memory.reset() was never called.
                if bool(kwargs.get("reset", False)) or any(int(t) == 0 for t in timesteps):
                    self.memory.reset()
                # [Geo-MemoryVLA] pooled cog token into memory (fixed size) — see _pool_cog_token.
                m_geo, m_sem = self.memory.process(geo, sem_cog, episode_ids, timesteps)
            imag = None
            if self.use_imag:
                # [Geo-MemoryVLA] S2: inference samples carry no image_window (no rolling buffer),
                # so allow a degenerate static window instead of crashing the eval server.
                if getattr(self, "_backbone", "vggt_world") == "d4rt":
                    # [D4RT-WorldState] agentview window; replicate-pads current frame at eval.
                    window = _build_d4rt_window(examples, device=geo.device, win_len=self._d4rt_win_len)
                    window = self._cast_to_vggt_dtype(window)
                else:
                    window = self._build_image_window(examples, device=geo.device, allow_degenerate=True)
                imag = self.imaginer.imagine_tokens(window, forecast_frames=self.imag_horizon)
            cond, mask = self._assemble(geo, sem, m_geo, m_sem, imag)

        state = [e["state"] for e in examples] if "state" in examples[0] else None
        state_t = torch.from_numpy(np.array(state)).to(cond.device, dtype=cond.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(cond, state_t, encoder_attention_mask=mask)
        return {"normalized_actions": pred.detach().cpu().numpy()}
