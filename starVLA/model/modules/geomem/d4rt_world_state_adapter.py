# starVLA/model/modules/geomem/d4rt_world_state_adapter.py
# [D4RT-WorldState] Thin starVLA-facing wrapper over the vendored Open-D4RT encoder.
# Mirrors WorldStateAdapter: encode(window)->D4RTState, flatten(state)->[B,N,C].
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from starVLA.model.modules.d4rt.loader import build_d4rt_model


@dataclass
class D4RTState:
    memory: torch.Tensor   # [B, N, C]  Global Scene Representation (D4RT "memory")
    video: torch.Tensor    # [B, T, 3, 256, 256]  retained so the imaginer can query the decoder

    def flatten(self) -> torch.Tensor:
        # D4RT memory is already [B, N, C] (token stream), so flatten is identity.
        return self.memory


class D4RTWorldStateAdapter(nn.Module):
    def __init__(self, model_yaml: str, ckpt_path: str | None = None) -> None:
        super().__init__()
        self.model = build_d4rt_model(model_yaml, ckpt_path)
        for p in self.model.parameters():            # frozen encoder (spec R4)
            p.requires_grad_(False)
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
        return state.flatten()
