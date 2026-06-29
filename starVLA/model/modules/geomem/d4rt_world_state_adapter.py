# starVLA/model/modules/geomem/d4rt_world_state_adapter.py
# [D4RT-WorldState] Thin starVLA-facing wrapper over the vendored Open-D4RT encoder.
# Mirrors WorldStateAdapter: encode(window)->D4RTState, flatten(state)->[B,N,C].
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.d4rt.loader import build_d4rt_model


def pool_d4rt_tokens(memory: torch.Tensor, k: int, spatial: int = 256) -> torch.Tensor:
    """Compress D4RT memory [B, N, C] along the SPATIAL axis, keeping the TEMPORAL axis.

    D4RT memory is N = T_steps * spatial (+ leading global/aspect tokens). Adaptive-pool the
    `spatial` patches of each time-step to `k`, so N drops T_steps*spatial -> T_steps*k while the
    temporal structure (the world-model value) is preserved. Leading remainder tokens (N % spatial,
    e.g. the +1 aspect token) are kept verbatim. No-op if N <= spatial (already short).
    """
    B, N, C = memory.shape
    if N <= spatial or k >= spatial:
        return memory
    lead = N % spatial                       # leading non-grid tokens (e.g. +1 aspect/global)
    grid = memory[:, lead:, :]               # [B, T_steps*spatial, C]
    t_steps = grid.shape[1] // spatial
    grid = grid.reshape(B, t_steps, spatial, C)              # [B, T, S, C]
    # adaptive avg-pool the spatial axis S -> k (per time-step), dim-preserving.
    grid = grid.permute(0, 1, 3, 2).reshape(B * t_steps, C, spatial)   # [B*T, C, S]
    grid = F.adaptive_avg_pool1d(grid, k)                    # [B*T, C, k]
    grid = grid.reshape(B, t_steps, C, k).permute(0, 1, 3, 2).reshape(B, t_steps * k, C)
    if lead:
        return torch.cat([memory[:, :lead, :], grid], dim=1)  # [B, lead + T*k, C]
    return grid


@dataclass
class D4RTState:
    memory: torch.Tensor   # [B, N, C]  Global Scene Representation (D4RT "memory")
    video: torch.Tensor    # [B, T, 3, 256, 256]  retained so the imaginer can query the decoder

    def flatten(self, pool_k: int = 0) -> torch.Tensor:
        # D4RT memory is [B, N, C]. pool_k>0 compresses spatial patches/time-step to bound the
        # GR00T cross-attention condition (6145 tokens @ 48 frames otherwise -> OOM). 0 = identity.
        if pool_k and pool_k > 0:
            return pool_d4rt_tokens(self.memory, k=pool_k)
        return self.memory


class D4RTWorldStateAdapter(nn.Module):
    def __init__(self, model_yaml: str, ckpt_path: str | None = None,
                 geo_pool_tokens: int = 8) -> None:
        super().__init__()
        self.model = build_d4rt_model(model_yaml, ckpt_path)
        for p in self.model.parameters():            # frozen encoder (spec R4)
            p.requires_grad_(False)
        # [D4RT-WorldState] spatial patches/time-step kept after pooling the geo condition.
        # 0 = no pooling (the old, OOM-prone behavior). Default 8 -> ~T*8 tokens.
        self.geo_pool_tokens = int(geo_pool_tokens)
        # [D4RT-WorldState] Output channel C == dec_hidden, which the heads were built with
        # (D4RTHeads.xyz_head: Linear(dec_hidden, 3)). Read it directly so hidden_size is
        # known at construction — the orchestrator reads geo_dim before any forward pass.
        self._hidden_size = int(self.model.heads.xyz_head.in_features)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def encode(self, window: torch.Tensor) -> D4RTState:
        """window: [B, T, 3, 256, 256] in [0,1] (agentview-only temporal window, spec §5)."""
        window = window.to(self.dtype)
        memory = self.model.encode_video(video=window, aspect_ratio=None)  # [B, N, C]
        return D4RTState(memory=memory, video=window)

    def flatten(self, state: D4RTState) -> torch.Tensor:
        return state.flatten(pool_k=self.geo_pool_tokens)
