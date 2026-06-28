# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/backbone.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from vggt.models.aggregator import slice_expand_and_flatten
from vggt.models.vggt import VGGT

from .state import GeometryState


class FrozenVGGTBackbone(nn.Module):
    def __init__(self, vggt_model: VGGT):
        super().__init__()
        self.vggt = vggt_model
        self.aggregator = vggt_model.aggregator
        for param in self.vggt.parameters():
            param.requires_grad = False
        self.vggt.eval()

    @classmethod
    def from_pretrained(cls, repo_id: str = "facebook/VGGT-1B") -> "FrozenVGGTBackbone":
        return cls(VGGT.from_pretrained(repo_id))

    def _get_positions(self, bsz: int, frames: int, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        aggregator = self.aggregator
        if aggregator.rope is None:
            return None
        pos = aggregator.position_getter(bsz * frames, height // aggregator.patch_size, width // aggregator.patch_size, device=device)
        if aggregator.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(bsz * frames, aggregator.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        return pos

    def _prepare_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        aggregator = self.aggregator
        bsz, frames, channels, height, width = images.shape
        images = (images - aggregator._resnet_mean) / aggregator._resnet_std
        images = images.view(bsz * frames, channels, height, width)
        patch_tokens = aggregator.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        camera_token = slice_expand_and_flatten(aggregator.camera_token, bsz, frames)
        register_token = slice_expand_and_flatten(aggregator.register_token, bsz, frames)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        pos = self._get_positions(bsz, frames, height, width, images.device)
        return tokens, pos, height // aggregator.patch_size, width // aggregator.patch_size

    def _run_layers(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor | None,
        bsz: int,
        frames: int,
        start_layer: int,
        end_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor | None]:
        if tokens.ndim == 4:
            patch_count = tokens.shape[2]
            dim = tokens.shape[-1]
        else:
            patch_count = tokens.shape[1]
            dim = tokens.shape[-1]
        outputs: List[torch.Tensor] = []
        frame_idx = start_layer
        global_idx = start_layer
        last_frame = None
        last_global = None
        for _ in range(start_layer, end_layer):
            tokens, frame_idx, frame_intermediates = self.aggregator._process_frame_attention(tokens, bsz, frames, patch_count, dim, frame_idx, pos=pos)
            tokens, global_idx, global_intermediates = self.aggregator._process_global_attention(tokens, bsz, frames, patch_count, dim, global_idx, pos=pos)
            last_frame = frame_intermediates[-1]
            last_global = global_intermediates[-1]
            outputs.append(torch.cat([last_frame, last_global], dim=-1))
        if last_frame is None or last_global is None:
            raise RuntimeError("No layers were executed.")
        global_hidden = tokens.view(bsz, frames, patch_count, dim) if tokens.ndim == 3 else tokens
        return last_frame, global_hidden, outputs, pos

    @torch.no_grad()
    def encode_states(self, images: torch.Tensor) -> GeometryState:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        bsz, frames, _, height, width = images.shape
        tokens, pos, patch_h, patch_w = self._prepare_tokens(images)
        frame_tokens, global_tokens, _, _ = self._run_layers(tokens, pos, bsz, frames, start_layer=0, end_layer=5)
        frame_ids = torch.arange(frames, device=images.device).view(1, frames).expand(bsz, -1)
        return GeometryState(
            frame_tokens=frame_tokens,
            global_tokens=global_tokens,
            patch_start_idx=self.aggregator.patch_start_idx,
            patch_grid=(patch_h, patch_w),
            frame_ids=frame_ids,
            image_hw=(height, width),
        )

    @torch.no_grad()
    def decode_geometry(self, state: GeometryState, images: torch.Tensor | None = None) -> dict:
        if images is not None and images.ndim == 4:
            images = images.unsqueeze(0)

        if images is None:
            if state.image_hw is None:
                raise ValueError("GeometryState.image_hw is required when decoding without images.")
            height, width = state.image_hw
            images = torch.zeros(
                state.batch_size,
                state.num_frames,
                3,
                height,
                width,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            height, width = images.shape[-2:]
            if images.shape[1] != state.num_frames:
                images = torch.zeros(
                    state.batch_size,
                    state.num_frames,
                    3,
                    height,
                    width,
                    device=state.device,
                    dtype=state.dtype,
                )

        bsz, frames = images.shape[:2]
        pos = self._get_positions(bsz, frames, height, width, images.device)
        _, _, outputs, _ = self._run_layers(state.global_tokens, pos, bsz, frames, start_layer=5, end_layer=self.aggregator.depth)
        aggregated_tokens_list: List[torch.Tensor] = [state.features for _ in range(5)] + outputs

        out = {
            "aggregated_tokens_list": aggregated_tokens_list,
            "patch_start_idx": state.patch_start_idx,
        }
        if self.vggt.camera_head is not None:
            out["pose_enc_list"] = self.vggt.camera_head(aggregated_tokens_list)
            out["pose_enc"] = out["pose_enc_list"][-1]
        if self.vggt.depth_head is not None:
            depth, depth_conf = self.vggt.depth_head(aggregated_tokens_list, images, state.patch_start_idx)
            out["depth"] = depth
            out["depth_conf"] = depth_conf
        if self.vggt.point_head is not None:
            world_points, world_points_conf = self.vggt.point_head(aggregated_tokens_list, images, state.patch_start_idx)
            out["world_points"] = world_points
            out["world_points_conf"] = world_points_conf
        return out
