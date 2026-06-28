# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/scheduler.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch


def linear_ramp(start: float, end: float, where: float) -> float:
    where = min(max(where, 0.0), 1.0)
    return start + (end - start) * where


def sample_tau(batch_size: int, device: torch.device, eps: float = 1e-4) -> torch.Tensor:
    return torch.rand(batch_size, device=device) * (1.0 - eps)


def sample_tau_mid(batch_size: int, device: torch.device, eps: float = 1e-4) -> torch.Tensor:
    return eps + (1.0 - 2.0 * eps) * torch.rand(batch_size, device=device)
