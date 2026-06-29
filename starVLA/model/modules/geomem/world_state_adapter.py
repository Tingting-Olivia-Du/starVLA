# starVLA/model/modules/geomem/world_state_adapter.py
# [Geo-MemoryVLA] Thin starVLA-facing wrapper over the vendored VGGT-World backbone.
# Turns images into a GeometryState world state and flattens it for downstream
# memory / condition assembly. Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.vggt_world.backbone import FrozenVGGTBackbone


class WorldStateAdapter(nn.Module):
    HIDDEN_SIZE = 1024

    def __init__(self, pretrained_vggt_repo: str = "facebook/VGGT-1B") -> None:
        super().__init__()
        self.backbone = FrozenVGGTBackbone.from_pretrained(pretrained_vggt_repo)

    @property
    def hidden_size(self) -> int:
        return self.HIDDEN_SIZE

    @property
    def dtype(self) -> torch.dtype:
        # [Geo-MemoryVLA] Actual dtype of the (frozen) VGGT params. Pixel inputs must match it
        # so the conv works without relying on an ambient autocast (eval/predict_action has none).
        return next(self.backbone.parameters()).dtype

    def encode(self, images: torch.Tensor):
        """images: [B, num_views(*frames), 3, H, W] -> GeometryState (frozen)."""
        return self.backbone.encode_states(images)

    def flatten(self, state) -> torch.Tensor:
        """GeometryState -> [B, F*2*tokens, 1024]."""
        return state.flatten_streams()
