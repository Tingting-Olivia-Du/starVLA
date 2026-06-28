# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/flow_model.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch import Tensor

from .blocks import DoubleStreamBlock, SingleStreamBlock
from .rope3d import EmbedND3D
from .time_embed import MLPEmbedder, timestep_embedding


class LastLayer(nn.Module):
    """
    Final adaptive-norm output layer.
    Adapted from FLUX ``layers.py:242-253``.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class TemporalFlowTransformer(nn.Module):
    """
    Flow transformer for geometry world modeling.

    Architecture mirrors FLUX ``model.py:34-119``, adapted for asymmetric
    double-stream attention (VGGT-World paper Eq. 7) and 3-axis RoPE.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int | None = None,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        double_stream_depth: int = 8,
        single_stream_depth: int = 8,
        qkv_bias: bool = True,
        theta: int = 10_000,
        axes_dim: list[int] | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_dim} must be divisible by num_heads {num_heads}")
        head_dim = hidden_dim // num_heads
        axes_dim = axes_dim or [16, 24, 24]
        if sum(axes_dim) != head_dim:
            raise ValueError(f"Got axes_dim {axes_dim} (sum={sum(axes_dim)}) but expected head_dim {head_dim}")

        self.hidden_size = hidden_dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.pe_embedder = EmbedND3D(dim=head_dim, theta=theta, axes_dim=axes_dim)
        self.input_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.condition_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.cond_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_dim)

        self.double_stream = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(double_stream_depth)
            ]
        )

        self.single_stream = nn.ModuleList(
            [
                SingleStreamBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(single_stream_depth)
            ]
        )

        self.final_layer = LastLayer(hidden_dim, dim)

    def _forward_blocks(
        self,
        target: Tensor,
        cond: Tensor,
        vec: Tensor,
        pe_target: Tensor,
        pe_kv: Tensor,
    ) -> Tensor:
        """Run all transformer blocks. Separated for Euler loop reuse."""
        use_ckpt = self.gradient_checkpointing and self.training

        for block in self.double_stream:
            if use_ckpt:
                target = ckpt.checkpoint(block, target, cond, vec, pe_target, pe_kv, use_reentrant=False)
            else:
                target = block(img=target, cond=cond, vec=vec, pe_img=pe_target, pe_kv=pe_kv)

        for block in self.single_stream:
            if use_ckpt:
                target = ckpt.checkpoint(block, target, vec, pe_target, use_reentrant=False)
            else:
                target = block(target, vec=vec, pe=pe_target)

        return self.final_layer(target, vec)

    def forward(
        self,
        target_noisy: Tensor,
        *,
        tau: Tensor,
        condition: Tensor,
        target_pos: Tensor,
        condition_pos: Tensor,
        _precomputed: Optional[dict] = None,
    ) -> Tensor:
        if target_noisy.ndim != 3 or condition.ndim != 3:
            raise ValueError("Input target_noisy and condition tensors must have 3 dimensions.")

        target = self.input_proj(target_noisy)

        if _precomputed is not None:
            cond = _precomputed["cond"]
            pe_target = _precomputed["pe_target"]
            pe_kv = _precomputed["pe_kv"]
        else:
            cond = self.cond_norm(self.condition_proj(condition))
            pe_target = self.pe_embedder(target_pos)
            pe_condition = self.pe_embedder(condition_pos)
            pe_kv = torch.cat((pe_target, pe_condition), dim=2)

        vec = self.time_in(timestep_embedding(tau, 256))

        return self._forward_blocks(target, cond, vec, pe_target, pe_kv)

    def precompute_condition(
        self,
        condition: Tensor,
        target_pos: Tensor,
        condition_pos: Tensor,
    ) -> dict:
        """Pre-compute condition projection and RoPE for Euler loop reuse."""
        cond = self.cond_norm(self.condition_proj(condition))
        pe_target = self.pe_embedder(target_pos)
        pe_condition = self.pe_embedder(condition_pos)
        pe_kv = torch.cat((pe_target, pe_condition), dim=2)
        return {"cond": cond, "pe_target": pe_target, "pe_kv": pe_kv}
