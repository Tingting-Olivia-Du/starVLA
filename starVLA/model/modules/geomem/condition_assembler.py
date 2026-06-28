# starVLA/model/modules/geomem/condition_assembler.py
# [Geo-MemoryVLA] Projects each stream (geometry/semantic/memory/imagination)
# to a shared dim and concatenates into the [B, L, H] condition the GR00T head
# consumes. Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionAssembler(nn.Module):
    def __init__(self, stream_dims: dict[str, int], out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.ModuleDict({name: nn.Linear(d, out_dim) for name, d in stream_dims.items()})

    def forward(self, streams: dict[str, torch.Tensor]):
        parts, masks = [], []
        for name, x in streams.items():
            if x is None:
                continue
            p = self.proj[name](x)  # [B, N_i, out_dim]
            parts.append(p)
            masks.append(torch.ones(p.shape[0], p.shape[1], dtype=torch.bool, device=p.device))
        cond = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        return cond, mask
