# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/blocks.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .rope3d import attention, attention_asymmetric
from .time_embed import Modulation, QKNorm, RMSNorm


class SelfAttention(nn.Module):
    """
    Self-attention with fused QKV projection and RMSNorm-based QK normalization.
    Matches FLUX ``layers.py:87-103`` exactly.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


class DoubleStreamBlock(nn.Module):
    """
    Asymmetric double-stream block.

    Follows FLUX ``layers.py:129-191`` for the target (img) stream but adapts
    the condition (txt) stream to be read-only: no modulation, no Q, no update,
    no FFN.  This matches VGGT-World paper Eq. 7: the condition ``c_i`` is kept
    clean while the target cross-attends to it.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Target stream — mirrors FLUX img stream (lines 136-145)
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Condition stream — K,V only (no Q, no modulation, no FFN, no update)
        # Note: condition LayerNorm is applied once in flow_model.py, not per-block
        self.cond_kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.cond_k_norm = RMSNorm(hidden_size // num_heads)

    def forward(
        self,
        img: Tensor,
        cond: Tensor,
        vec: Tensor,
        pe_img: Tensor,
        pe_kv: Tensor,
    ) -> Tensor:
        img_mod1, img_mod2 = self.img_mod(vec)

        # prepare target for attention (mirrors FLUX lines 162-167)
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare condition: K,V only, no modulation (paper Eq. 7)
        # cond is already LayerNorm'd once in flow_model.py
        cond_kv = self.cond_kv(cond)
        cond_k, cond_v = rearrange(cond_kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        cond_k = self.cond_k_norm(cond_k).to(cond_v)

        # joint attention (mirrors FLUX lines 176-182)
        q = img_q
        k = torch.cat((img_k, cond_k), dim=2)
        v = torch.cat((img_v, cond_v), dim=2)
        attn = attention_asymmetric(q, k, v, pe_q=pe_img, pe_kv=pe_kv)

        # target update only (mirrors FLUX lines 184-186)
        img = img + img_mod1.gate * self.img_attn.proj(attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        return img


class SingleStreamBlock(nn.Module):
    """
    Fused parallel attention + MLP block.
    Matches FLUX ``layers.py:194-239`` exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output
