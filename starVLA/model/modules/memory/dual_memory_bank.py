# starVLA/model/modules/memory/dual_memory_bank.py
# [Geo-MemoryVLA] Dual (geometry + semantic) memory: VGGT latent in the geometric
# bank, Qwen3-VL cognitive token in the semantic bank — the perceptual/cognitive
# split from MemoryVLA, moved into a dual-latent space. Part of the Geo-MemoryVLA
# architecture — see docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch.nn as nn

from .memory_bank import MemoryBank


class DualMemoryBank(nn.Module):
    def __init__(self, geo_dim: int, sem_dim: int, **bank_kwargs):
        super().__init__()
        self.geo = MemoryBank(token_dim=geo_dim, **bank_kwargs)
        self.sem = MemoryBank(token_dim=sem_dim, **bank_kwargs)

    def reset(self) -> None:
        self.geo.reset()
        self.sem.reset()

    def process(self, geo, sem, episode_ids, timesteps):
        m_geo = self.geo.process_batch(geo, episode_ids, timesteps) if geo is not None else None
        m_sem = self.sem.process_batch(sem, episode_ids, timesteps) if sem is not None else None
        return m_geo, m_sem
