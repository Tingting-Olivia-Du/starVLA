# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/solver.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch


def clean_to_velocity(z_tau: torch.Tensor, pred_clean: torch.Tensor, tau: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    while tau.ndim < z_tau.ndim:
        tau = tau.unsqueeze(-1)
    denom = tau.clamp_min(eps)
    return (z_tau - pred_clean) / denom


def euler_rollout(
    model,
    noisy_state: torch.Tensor,
    condition: torch.Tensor,
    target_pos: torch.Tensor,
    condition_pos: torch.Tensor,
    tau_start: torch.Tensor,
    tau_end: torch.Tensor,
    steps: int,
):
    # Pre-compute condition projection and RoPE once for the entire rollout
    if hasattr(model, "precompute_condition"):
        precomputed = model.precompute_condition(condition, target_pos, condition_pos)
    else:
        precomputed = None

    state = noisy_state
    tau = tau_start
    dt = (tau_end - tau_start) / max(steps, 1)
    tau_step = dt
    for _ in range(steps):
        pred_clean = model(
            state,
            tau=tau,
            condition=condition,
            target_pos=target_pos,
            condition_pos=condition_pos,
            _precomputed=precomputed,
        )
        velocity = clean_to_velocity(state, pred_clean, tau)
        step = dt
        while step.ndim < velocity.ndim:
            step = step.unsqueeze(-1)
        state = state + step * velocity
        tau = tau + tau_step
    return state
