# Copyright 2026 3DVLA-Stage2. MIT License.
# Knowledge-Insulated Qwen VLA: FAST next-token prediction for the backbone
# + GR00T flow-matching expert behind a stop-gradient.
#
# Design doc: 3dvla-stage0/docs/stage2-ki-design.md. Key choices:
#   * NO "-Action" vocab resize. The plain Qwen3-VL vocabulary/lm_head are
#     untouched: FAST action tokens live in a SEPARATE 2049-way embedding
#     (2048 FAST ids + 1 STOP) injected at reserved placeholder positions via
#     a forward hook on embed_tokens, and are read out through a SEPARATE
#     linear head. Softmax insulation: text-token probabilities are bit-equal
#     to the grafted Stage-1 checkpoint (unit test 3).
#   * Flow expert consumes last_hidden.detach() -> zero gradient into the
#     backbone (unit test 1), and its encoder attention mask zeroes the action
#     slots -> the expert conditions on the vision/language prefix only, never
#     on teacher-forced ground-truth action tokens.

"""
QwenFastGR00T_KI Framework

Training forward (vla_data batches):
  1. FAST-encode action chunk -> id sequence per sample.
  2. Build standard QwenVL prompt inputs (images + instruction), then append
     k+1 placeholder slots per sample (k action ids + STOP slot).
  3. embed_tokens hook swaps placeholder rows for action_embed(fast_ids).
  4. Backbone CE over slot positions through action_lm_head (separate head).
  5. Flow-matching loss on detached hidden states, prompt-only encoder mask.
  loss = w_fast * fast_ce + w_flow * flow_loss

Inference:
  predict_action      -> flow expert (deployment form, pi0.5-KI style).
  predict_action_fast -> autoregressive FAST decode (diagnostic; no KV cache).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model as get_flow_head,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

FAST_VOCAB = 2048          # FAST+ id space
STOP_ID = FAST_VOCAB       # extra class: end-of-action-sequence
# Reserved input-side marker for action slots. Row 151669 is the first of the
# 267 allocated-but-unused vocabulary rows of plain Qwen3-VL (vocab 151936);
# the tokenizer never emits it, and the hook REPLACES its embedding output, so
# its (untrained) weight row never reaches the transformer.
PLACEHOLDER_ID = 151669


@dataclass
class QwenFastGR00TKIDefaultConfig:
    """Defaults; YAML ``framework:`` keys override, extras preserved."""

    name: str = "QwenFastGR00T_KI"

    qwenvl: dict = field(
        default_factory=lambda: {
            # PLAIN instruct checkpoint (Stage-1 variants are grafted onto
            # this architecture) — NOT the resized "-Action" checkpoint.
            "base_vlm": "Qwen/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
        }
    )

    # Flow expert config — mirrors QwenGR00T so the expert is NOT a variable.
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # KI-specific knobs
    ki: dict = field(
        default_factory=lambda: {
            "w_fast": 1.0,        # weight of backbone FAST CE
            "w_flow": 1.0,        # weight of flow-matching loss
            "max_fast_len": 64,   # safety cap on FAST ids per chunk
            # HF hub id or absolute local path (the repo-relative default of
            # fast_ActionHeader only resolves from the starVLA repo root)
            "fast_tokenizer_path": "physical-intelligence/fast",
            # Deployment-contract focal unification (plan §3.1: applied
            # VERBATIM in VLA preprocessing). Per-view fx in PIXEL units of
            # the incoming frames, listed in the data config's video_keys
            # order. LIBERO lerobot 256x256: agentview fovy 45deg -> fx
            # 579.41*(256/480)=309.02; wrist fovy ~75deg -> 312.77*(256/480)
            # =166.81. Empty list = no unification (raw frames).
            "view_fx": [],
            "f_target": 600.0,
            # HF gradient checkpointing on the backbone (the trainer yaml key
            # is dead in the cotrain script; this knob actually applies it)
            "gradient_checkpointing": False,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenFastGR00T_KI")
class Qwenvl_FastGR00T_KI(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenFastGR00TKIDefaultConfig, config)
        assert "-Action" not in self.config.framework.qwenvl.get("base_vlm", ""), (
            "KI framework requires the PLAIN Qwen3-VL checkpoint; the resized "
            "'-Action' vocab breaks softmax insulation (design doc §2)")

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        hidden = self.qwen_vl_interface.model.config.hidden_size

        # flow expert (frozen recipe, not a studied variable)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden
        self.flow_action_model: FlowmatchingActionHead = get_flow_head(config=self.config)

        # FAST tokenizer (parameter-free)
        self.fast_action_model = Fast_Action_Tokenizer(
            fast_tokenizer_name=self.config.framework.ki.fast_tokenizer_path)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.fast_action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.fast_action_model.fast_tokenizer.action_dim = (
            self.config.framework.action_model.action_dim)

        # KI insulation modules (bf16 to match backbone activations)
        self.action_embed = nn.Embedding(FAST_VOCAB + 1, hidden, dtype=torch.bfloat16)
        self.action_lm_head = nn.Linear(hidden, FAST_VOCAB + 1, bias=False,
                                        dtype=torch.bfloat16)

        self.w_fast = float(self.config.framework.ki.w_fast)
        self.w_flow = float(self.config.framework.ki.w_flow)
        self.max_fast_len = int(self.config.framework.ki.max_fast_len)

        # Injection hook: replace placeholder rows of embed_tokens output with
        # action embeddings staged in _pending_slot_ids (set per forward).
        self._pending_slot_ids: Optional[torch.Tensor] = None  # flat [N] long
        emb = self.qwen_vl_interface.model.get_input_embeddings()
        emb.register_forward_hook(self._inject_action_embeds)

        # Deployment-contract focal unification (per-view uniform resize to
        # f_target; dynamic-resolution backbone -> no canvas, coords invariant)
        self.view_fx = [float(x) for x in self.config.framework.ki.get("view_fx", [])]
        self.f_target = float(self.config.framework.ki.get("f_target", 600.0))

        if self.config.framework.ki.get("gradient_checkpointing", False):
            self.qwen_vl_interface.model.gradient_checkpointing_enable()
            logger.info("[KI] backbone gradient checkpointing ON")

        # Optional Stage-1 graft: load a trained VLM checkpoint into the
        # backbone (keys are framework-shaped: qwen_vl_interface.model.*).
        stage1_ckpt = self.config.framework.qwenvl.get("stage1_ckpt", None)
        if stage1_ckpt:
            from safetensors.torch import load_file
            sd = load_file(stage1_ckpt)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            assert not unexpected, f"stage1 graft: unexpected keys {unexpected[:5]}"
            bad = [k for k in missing
                   if not (k.startswith(("action_embed", "action_lm_head",
                                         "flow_action_model", "fast_action_model"))
                           or "rotary_emb" in k or "position_ids" in k)]
            assert not bad, f"stage1 graft: missing backbone keys {bad[:5]}"
            logger.info(f"[KI] grafted Stage-1 backbone from {stage1_ckpt}")

        # Vocab logits are never consumed here (insulated head instead) —
        # truncate the lm_head projection to 1 position when supported
        # (saves ~[B,L,152K] bf16 per forward).
        import inspect
        self._logits_kw = None
        sig = inspect.signature(self.qwen_vl_interface.model.forward)
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            if kw in sig.parameters:
                self._logits_kw = kw
                break

    # ------------------------------------------------------------------ hook
    def _inject_action_embeds(self, module, inputs, output):
        if self._pending_slot_ids is None:
            return output
        input_ids = inputs[0]
        mask = input_ids == PLACEHOLDER_ID
        n = int(mask.sum())
        if n == 0:
            return output
        assert n == self._pending_slot_ids.numel(), (
            f"placeholder count {n} != staged slot ids {self._pending_slot_ids.numel()}")
        inj = self.action_embed(self._pending_slot_ids.to(input_ids.device))
        output = output.clone()
        output[mask] = inj.to(output.dtype)
        return output

    # ---------------------------------------------------------- focal unify
    def _unify_views(self, batch_images):
        """Apply the deployment-contract focal unification: per-view uniform
        resize so fx -> f_target (dynamic-resolution backbone, no canvas)."""
        if not self.view_fx:
            return batch_images
        from PIL import Image
        out = []
        for imgs in batch_images:
            assert len(imgs) == len(self.view_fx), (
                f"{len(imgs)} views != {len(self.view_fx)} configured view_fx")
            row = []
            for im, fx in zip(imgs, self.view_fx):
                if not isinstance(im, Image.Image):
                    im = Image.fromarray(np.asarray(im))
                s = self.f_target / fx
                row.append(im.resize((max(28, round(im.size[0] * s)),
                                      max(28, round(im.size[1] * s))),
                                     Image.BICUBIC))
            out.append(row)
        return out

    # ------------------------------------------------------------- fast utils
    def _encode_fast(self, actions) -> List[List[int]]:
        ids_batch = self.fast_action_model.encoder_action2fastoken(actions)
        out = []
        for ids in ids_batch:
            ids = list(map(int, ids))[: self.max_fast_len]
            assert all(0 <= t < FAST_VOCAB for t in ids), f"FAST id out of range: {ids}"
            out.append(ids)
        return out

    def _append_slots(self, qwen_inputs, slot_ids_batch: List[List[int]]):
        """Append per-sample [k+1] placeholder slots (right side; prompt is
        left-padded) and stage flat slot ids for the hook. Returns extended
        (input_ids, attention_mask, fast_labels)."""
        input_ids = qwen_inputs["input_ids"]
        attn = qwen_inputs["attention_mask"]
        B, Lp = input_ids.shape
        pad_id = self.qwen_vl_interface.processor.tokenizer.pad_token_id
        k_max = max(len(s) for s in slot_ids_batch)

        ext_ids = input_ids.new_full((B, k_max), pad_id)
        ext_attn = attn.new_zeros((B, k_max))
        fast_labels = input_ids.new_full((B, Lp + k_max), -100)
        flat = []
        for i, slots in enumerate(slot_ids_batch):
            k = len(slots)
            ext_ids[i, :k] = PLACEHOLDER_ID
            ext_attn[i, :k] = 1
            fast_labels[i, Lp: Lp + k] = torch.tensor(slots, device=input_ids.device)
            flat.extend(slots)
        self._pending_slot_ids = torch.tensor(flat, dtype=torch.long,
                                              device=input_ids.device)
        return (torch.cat([input_ids, ext_ids], dim=1),
                torch.cat([attn, ext_attn], dim=1), fast_labels)

    # ---------------------------------------------------------------- forward
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = self._unify_views([ex["image"] for ex in examples])
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        # [3DVLA] fail-fast input guard: a single NaN (e.g. zero-range
        # normalization stats) silently poisons ALL weights within one step —
        # crash loudly instead (bridge state[6] q01==q99 lesson, 2026-07-12)
        _a = np.asarray(actions, dtype=np.float64)
        assert np.isfinite(_a).all(), (
            f"[KI] non-finite ACTIONS in batch (dims with NaN/inf: "
            f"{sorted(set(np.argwhere(~np.isfinite(_a))[:, -1].tolist()))}) — "
            "check normalization stats for zero-range dims")
        if state is not None:
            _s = np.asarray(state, dtype=np.float64)
            assert np.isfinite(_s).all(), (
                f"[KI] non-finite STATE in batch (dims: "
                f"{sorted(set(np.argwhere(~np.isfinite(_s))[:, -1].tolist()))}) — "
                "check normalization stats for zero-range dims")

        # [3DVLA] image guard: corrupt video decode can yield NaN frames
        for bi, imgs in enumerate(batch_images):
            for vi, im in enumerate(imgs):
                arr = np.asarray(im, dtype=np.float32)
                assert np.isfinite(arr).all(), (
                    f"[KI] non-finite IMAGE batch[{bi}] view[{vi}] "
                    f"(episode {examples[bi].get('episode_id')}) — corrupt video?")

        fast_ids = self._encode_fast(actions)
        slot_ids = [ids + [STOP_ID] for ids in fast_ids]  # input==label symmetry

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions)
        prompt_attn = qwen_inputs["attention_mask"].clone()  # prompt-only view
        input_ids, attn, fast_labels = self._append_slots(qwen_inputs, slot_ids)
        qwen_inputs["input_ids"], qwen_inputs["attention_mask"] = input_ids, attn

        extra = {self._logits_kw: 1} if self._logits_kw else {}
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.qwen_vl_interface(
                    **qwen_inputs, output_attentions=False,
                    output_hidden_states=True, return_dict=True, **extra)
        finally:
            self._pending_slot_ids = None
        h = outputs.hidden_states[-1]  # [B, L, H]

        # --- backbone FAST CE through the separate head (softmax-insulated)
        logits = self.action_lm_head(h[:, :-1])           # predict token at p from p-1
        targets = fast_labels[:, 1:]
        sel = targets != -100
        fast_ce = F.cross_entropy(logits[sel].float(), targets[sel])

        # --- flow expert behind stop-grad; encoder sees the PROMPT only
        # (action slots masked out -> no ground-truth-action leakage)
        B, Lp = prompt_attn.shape
        flow_mask = torch.zeros_like(attn)
        flow_mask[:, :Lp] = prompt_attn
        rep = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        with torch.autocast("cuda", dtype=torch.float32):
            acts = torch.tensor(np.array(actions), device=h.device, dtype=h.dtype)
            acts_target = acts[:, -self.action_horizon:, :].repeat(rep, 1, 1)
            h_flow = h.detach().repeat(rep, 1, 1)
            mask_rep = flow_mask.repeat(rep, 1).to(dtype=torch.bool)
            state_rep = None
            if state is not None:
                st = torch.tensor(np.array(state), device=h.device, dtype=h.dtype)
                state_rep = st.repeat(rep, 1, 1)
            flow_loss = self.flow_action_model(
                h_flow, acts_target, state_rep, encoder_attention_mask=mask_rep)

        total = self.w_fast * fast_ce + self.w_flow * flow_loss
        # [3DVLA] output guard: forensic report + dump, then crash loudly
        if not torch.isfinite(total):
            import os as _os
            first_bad_layer = None
            for li, hh in enumerate(outputs.hidden_states):
                if not torch.isfinite(hh).all():
                    first_bad_layer = li
                    break
            pv = qwen_inputs.get("pixel_values")
            pv_stats = (f"pixel_values finite={bool(torch.isfinite(pv).all())} "
                        f"min={float(pv.min()):.3f} max={float(pv.max()):.3f}"
                        if pv is not None else "no pixel_values")
            emb_w = self.qwen_vl_interface.model.get_input_embeddings().weight
            dump = f"/workspace/tingting/3dvla-checkpoints/ki_nan_batch_{_os.getpid()}.pt"
            torch.save({"examples": examples,
                        "input_ids": qwen_inputs["input_ids"].cpu(),
                        "attention_mask": qwen_inputs["attention_mask"].cpu(),
                        "fast_ce": float(fast_ce), "flow_loss": float(flow_loss)}, dump)
            raise RuntimeError(
                f"[KI] non-finite loss (fast_ce={float(fast_ce)}, flow={float(flow_loss)}); "
                f"FIRST NaN at hidden_states[{first_bad_layer}] of {len(outputs.hidden_states)}; "
                f"{pv_stats}; embed_tokens finite={bool(torch.isfinite(emb_w).all())}; "
                f"action_embed finite={bool(torch.isfinite(self.action_embed.weight).all())}; "
                f"dump={dump}")
        return {"action_loss": total, "fast_ce": fast_ce.detach(),
                "flow_loss": flow_loss.detach()}

    # -------------------------------------------------------------- inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Deployment form: flow expert on the prompt encoding (pi0.5-KI)."""
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = self._unify_views([ex["image"] for ex in examples])
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions)
        attn = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True)
        h = outputs.hidden_states[-1]
        st = (torch.from_numpy(np.array(state)).to(h.device, dtype=h.dtype)
              if state is not None else None)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.flow_action_model.predict_action(
                h, st, encoder_attention_mask=attn.to(dtype=torch.bool))
        return {"normalized_actions": pred.detach().cpu().numpy()}

    @torch.inference_mode()
    def predict_action_fast(self, examples: List[dict], **kwargs) -> dict:
        """Diagnostic: greedy AR FAST decode via the separate head. No KV
        cache (full re-forward per step) — probes/unit tests only."""
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = self._unify_views([ex["image"] for ex in examples])
        instructions = [ex["lang"] for ex in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions)
        B = qwen_inputs["input_ids"].shape[0]
        done = [False] * B
        decoded: List[List[int]] = [[] for _ in range(B)]
        for _ in range(self.max_fast_len):
            slot_ids = [d if d else [] for d in decoded]
            inputs = {k: v for k, v in qwen_inputs.items()}
            if any(slot_ids):
                input_ids, attn, _ = self._append_slots(
                    {"input_ids": qwen_inputs["input_ids"],
                     "attention_mask": qwen_inputs["attention_mask"]},
                    [s or [STOP_ID] for s in slot_ids])  # never empty rows
                inputs["input_ids"], inputs["attention_mask"] = input_ids, attn
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.qwen_vl_interface(
                        **inputs, output_hidden_states=True, return_dict=True)
            finally:
                self._pending_slot_ids = None
            h_last = outputs.hidden_states[-1][:, -1]      # [B, H]
            nxt = self.action_lm_head(h_last).argmax(-1)   # [B]
            for i in range(B):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t == STOP_ID:
                    done[i] = True
                else:
                    decoded[i].append(t)
            if all(done):
                break
        acts = self.fast_action_model.fast_tokenizer.decode(
            [d if d else [0] for d in decoded])
        return {"normalized_actions": acts, "fast_ids": decoded}
