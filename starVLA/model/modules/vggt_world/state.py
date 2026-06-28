# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/state.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class GeometryState:
    """
    Layer-4 decoder-compatible geometry state.

    The VGGT public architecture exposes both frame-stream and global-stream
    tokens at each layer index. The world state therefore keeps both streams
    explicitly at layer 4 while preserving the decoder entry state given by the
    global stream.
    """

    frame_tokens: torch.Tensor
    global_tokens: torch.Tensor
    patch_start_idx: int
    patch_grid: Tuple[int, int]
    frame_ids: Optional[torch.Tensor] = None
    image_hw: Optional[Tuple[int, int]] = None

    @property
    def device(self) -> torch.device:
        return self.frame_tokens.device

    @property
    def dtype(self) -> torch.dtype:
        return self.frame_tokens.dtype

    @property
    def batch_size(self) -> int:
        return self.frame_tokens.shape[0]

    @property
    def num_frames(self) -> int:
        return self.frame_tokens.shape[1]

    @property
    def num_tokens(self) -> int:
        return self.frame_tokens.shape[2]

    @property
    def dim(self) -> int:
        return self.frame_tokens.shape[-1]

    @property
    def features(self) -> torch.Tensor:
        return torch.cat([self.frame_tokens, self.global_tokens], dim=-1)

    def flatten_streams(self) -> torch.Tensor:
        stacked = torch.cat([self.frame_tokens, self.global_tokens], dim=2)
        return stacked.view(self.batch_size, self.num_frames * stacked.shape[2], self.dim)

    def slice_frames(self, start: int, end: int) -> "GeometryState":
        frame_ids = None if self.frame_ids is None else self.frame_ids[:, start:end]
        return GeometryState(
            frame_tokens=self.frame_tokens[:, start:end],
            global_tokens=self.global_tokens[:, start:end],
            patch_start_idx=self.patch_start_idx,
            patch_grid=self.patch_grid,
            frame_ids=frame_ids,
            image_hw=self.image_hw,
        )

    def clone(self) -> "GeometryState":
        frame_ids = None if self.frame_ids is None else self.frame_ids.clone()
        return GeometryState(
            frame_tokens=self.frame_tokens.clone(),
            global_tokens=self.global_tokens.clone(),
            patch_start_idx=self.patch_start_idx,
            patch_grid=self.patch_grid,
            frame_ids=frame_ids,
            image_hw=self.image_hw,
        )

    def detach(self) -> "GeometryState":
        frame_ids = None if self.frame_ids is None else self.frame_ids.detach()
        return GeometryState(
            frame_tokens=self.frame_tokens.detach(),
            global_tokens=self.global_tokens.detach(),
            patch_start_idx=self.patch_start_idx,
            patch_grid=self.patch_grid,
            frame_ids=frame_ids,
            image_hw=self.image_hw,
        )

    def with_updates(
        self,
        *,
        frame_tokens: Optional[torch.Tensor] = None,
        global_tokens: Optional[torch.Tensor] = None,
        frame_ids: Optional[torch.Tensor] = None,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> "GeometryState":
        return GeometryState(
            frame_tokens=self.frame_tokens if frame_tokens is None else frame_tokens,
            global_tokens=self.global_tokens if global_tokens is None else global_tokens,
            patch_start_idx=self.patch_start_idx,
            patch_grid=self.patch_grid,
            frame_ids=self.frame_ids if frame_ids is None else frame_ids,
            image_hw=self.image_hw if image_hw is None else image_hw,
        )


def concat_states(states: list[GeometryState], dim: int = 1) -> GeometryState:
    if not states:
        raise ValueError("Expected at least one state to concatenate.")

    reference = states[0]
    frame_ids = None
    if any(state.frame_ids is not None for state in states):
        frame_ids = torch.cat(
            [
                state.frame_ids
                if state.frame_ids is not None
                else torch.full(
                    (state.batch_size, state.num_frames),
                    -1,
                    device=state.device,
                    dtype=torch.long,
                )
                for state in states
            ],
            dim=dim,
        )

    return GeometryState(
        frame_tokens=torch.cat([state.frame_tokens for state in states], dim=dim),
        global_tokens=torch.cat([state.global_tokens for state in states], dim=dim),
        patch_start_idx=reference.patch_start_idx,
        patch_grid=reference.patch_grid,
        frame_ids=frame_ids,
        image_hw=reference.image_hw,
    )
