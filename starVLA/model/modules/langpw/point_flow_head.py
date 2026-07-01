# [LangPointWorld] Student point-flow head (C4, spec §3.1/§5): N learned point queries
# cross-attend the fused condition, an MLP maps each query to Hf x 3 future displacement.
# `flow_tokens` are the pre-projection query states, appended to the action-head condition.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch
import torch.nn as nn


class PointFlowHead(nn.Module):
    def __init__(self, cond_dim, n_points=256, flow_horizon=8, hidden=512, n_heads=8):
        super().__init__()
        self.n_points = n_points
        self.flow_horizon = flow_horizon
        self.hidden = hidden
        self.queries = nn.Parameter(torch.randn(n_points, hidden) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, flow_horizon * 3),
        )

    def forward(self, cond, encoder_attention_mask=None):
        """cond: [B,L,cond_dim]. Returns {"flow":[B,N,Hf,3], "flow_tokens":[B,N,hidden]}."""
        B = cond.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)          # [B,N,hidden]
        kv = self.cond_proj(cond)                                # [B,L,hidden]
        key_padding_mask = None
        if encoder_attention_mask is not None:
            # nn.MultiheadAttention expects True == "ignore"; our mask is True == "keep".
            key_padding_mask = ~encoder_attention_mask.bool()
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        tokens = self.norm(q + attn_out)                         # [B,N,hidden]
        flow = self.mlp(tokens).view(B, self.n_points, self.flow_horizon, 3)
        return {"flow": flow, "flow_tokens": tokens}
