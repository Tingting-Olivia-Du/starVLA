# starVLA/model/modules/memory/memory_bank.py
# [Geo-MemoryVLA] Dimension-agnostic temporal memory bank.
# Adapted from MemoryVLA (MIT): vla/memory_vla.py:CogMemBank — retrieval via
# cross-attention, gate fusion, ToMe consolidation. Part of the Geo-MemoryVLA
# architecture — see docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        a, _ = self.attn(self.norm(q), kv, kv)
        return q + a


class _GateFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, cur, mem):
        scale = torch.sigmoid(self.gate(torch.cat([cur, mem], dim=-1)))
        return scale * cur + (1.0 - scale) * mem


class MemoryBank(nn.Module):
    def __init__(
        self,
        token_dim: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        use_timestep_pe: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.mem_length = mem_length
        self.consolidate_type = consolidate_type
        self.use_timestep_pe = use_timestep_pe
        self.retrieval = nn.ModuleList([_CrossBlock(token_dim) for _ in range(retrieval_layers)])
        self.fusion = _GateFusion(token_dim) if fusion_type == "gate" else None
        # episode_id -> list of (timestep:int, feat:[N, D])
        self.bank: dict[int, list[tuple[int, torch.Tensor]]] = {}

    def reset(self) -> None:
        self.bank = {}

    def _timestep_pe(self, ts: int, n: int, d: int, device) -> torch.Tensor:
        if not self.use_timestep_pe:
            return torch.zeros(n, d, device=device)
        pos = torch.full((d,), float(ts), device=device)
        i = torch.arange(d, device=device)
        pe = torch.where(i % 2 == 0, torch.sin(pos / 10000 ** (i / d)), torch.cos(pos / 10000 ** (i / d)))
        return pe.unsqueeze(0).expand(n, d)

    def _consolidate(self, eid: int) -> None:
        hist = self.bank[eid]
        while len(hist) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[eid] = hist[-self.mem_length :]
                return
            # ToMe: merge the most-similar consecutive pair.
            best_i, best_sim = 0, -1e9
            for i in range(len(hist) - 1):
                a = hist[i][1].mean(0)
                b = hist[i + 1][1].mean(0)
                sim = F.cosine_similarity(a, b, dim=0).item()
                if sim > best_sim:
                    best_sim, best_i = sim, i
            ts_a, fa = hist[best_i]
            ts_b, fb = hist[best_i + 1]
            merged = (ts_b, 0.5 * (fa + fb))
            hist = hist[:best_i] + [merged] + hist[best_i + 2 :]
            self.bank[eid] = hist

    def process_batch(self, tokens: torch.Tensor, episode_ids, timesteps) -> torch.Tensor:
        b, n, d = tokens.shape
        outs = []
        for i in range(b):
            eid = int(episode_ids[i])
            ts = int(timesteps[i])
            cur = tokens[i : i + 1]  # [1, N, D]
            # [Geo-MemoryVLA] B2: order history by timestep before PE/retrieval/consolidation.
            # The mixture sampler can deliver an episode's steps out of order; memory is temporal,
            # so timestep PE and ToMe consecutive-pair merging must operate on time order.
            hist = sorted(self.bank.get(eid, []), key=lambda x: x[0])
            if hist:
                feats = []
                for h_ts, h_feat in hist:
                    pe = self._timestep_pe(h_ts, h_feat.shape[0], d, tokens.device)
                    feats.append(h_feat.to(tokens.device) + pe)
                mem = torch.cat(feats, dim=0).unsqueeze(0)  # [1, T*N, D]
                q = cur
                for blk in self.retrieval:
                    q = blk(q, mem)
                fused = self.fusion(cur, q) if self.fusion is not None else 0.5 * (cur + q)
            else:
                fused = cur
            outs.append(fused)
            # Write current observation into memory.
            self.bank.setdefault(eid, []).append((ts, tokens[i].detach().clone()))
            self._consolidate(eid)
        return torch.cat(outs, dim=0)
