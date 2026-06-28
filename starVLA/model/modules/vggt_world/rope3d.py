# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/rope3d.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    Per-axis rotary embedding as 2x2 rotation matrices.
    Matches FLUX ``math.py:15-22`` exactly.
    """
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to Q and K jointly (same sequence length).
    Matches FLUX ``math.py:25-30`` exactly.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def apply_rope_single(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary embeddings to a single tensor (for asymmetric attention)."""
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """
    Symmetric attention with RoPE (Q and K have same length).
    Matches FLUX ``math.py:6-12`` exactly.
    """
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def attention_asymmetric(q: Tensor, k: Tensor, v: Tensor, pe_q: Tensor, pe_kv: Tensor) -> Tensor:
    """Asymmetric attention with RoPE (Q shorter than K, separate pe tensors)."""
    q = apply_rope_single(q, pe_q)
    k = apply_rope_single(k, pe_kv)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


class EmbedND3D(nn.Module):
    """
    3-axis rotary position embedder.
    Adapts FLUX ``layers.py:11-25`` from 2 axes to 3 (time, y, x).
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)
