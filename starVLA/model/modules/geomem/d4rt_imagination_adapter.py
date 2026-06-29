# starVLA/model/modules/geomem/d4rt_imagination_adapter.py
# [D4RT-WorldState] Forecaster over the frozen D4RT encoder. Two subgoal types:
#   latent : a small head predicts a future-state token block F_hat (parity w/ VGGT-World z)
#   tracks : query the D4RT decoder at t_tgt>t for future 3D point positions (D4RT-native)
# Mirrors ImaginationAdapter method names (imagine_tokens / training_loss) so the
# GeoMemoryVLA call sites are unchanged. t_cam==t_tgt pinned via build_forecast_query (R2).
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTState
from tools.d4rt_forecast_probe import build_forecast_query


class LatentForecastHead(nn.Module):
    """Maps the observed Global Scene Representation to a future-state token block.

    Residual MLP: the future state is the current state plus a learned delta, so an
    untrained head starts near identity (a stable, near-static subgoal) and learns motion.
    """

    def __init__(self, dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:  # [B,N,C]->[B,N,C]
        return memory + self.net(memory)


class D4RTImaginationAdapter(nn.Module):
    def __init__(self, world_state, subgoal_type: str = "latent", horizon: int = 2,
                 track_grid: int = 8) -> None:
        super().__init__()
        assert subgoal_type in ("latent", "tracks"), f"bad subgoal_type: {subgoal_type!r}"
        self.subgoal_type = subgoal_type
        self.horizon = horizon
        self.track_grid = track_grid
        self._ws = world_state                       # shares the frozen D4RT model
        # latent head lazily sized to C on first use (C = world_state.hidden_size).
        self.latent_head: LatentForecastHead | None = None
        if subgoal_type == "latent" and hasattr(world_state, "hidden_size"):
            self._ensure_latent_head(int(world_state.hidden_size), torch.device("cpu"),
                                     torch.float32)

    def _ensure_latent_head(self, dim: int, device, dtype):
        if self.latent_head is None:
            self.latent_head = LatentForecastHead(dim, self.horizon).to(device=device, dtype=dtype)

    def _grid_uv(self, device) -> torch.Tensor:
        g = self.track_grid
        ys, xs = torch.meshgrid(torch.linspace(0, 1, g), torch.linspace(0, 1, g), indexing="ij")
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)  # [g*g,2]

    def imagine_tokens(self, state: D4RTState, forecast_frames: int) -> torch.Tensor:
        if self.subgoal_type == "latent":
            self._ensure_latent_head(state.memory.shape[-1], state.memory.device,
                                     state.memory.dtype)
            return self.latent_head(state.memory)            # [B,N,C] latent subgoal
        # tracks: query future 3D positions of a grid; concat xyz into a token block.
        model = self._ws.model
        T = state.video.shape[1]
        uv = self._grid_uv(state.video.device)
        outs = []
        for k in range(forecast_frames):
            q = build_forecast_query(uv, t_src=T - 1, t_tgt=T - 1 + k, device=state.video.device)
            xyz = model.decode_queries(video=state.video, query=q, memory=state.memory)["xyz_3d"]
            outs.append(xyz)                                  # [B, M, 3]
        return torch.cat(outs, dim=1)                         # [B, M*horizon, 3]

    def training_loss(self, state: D4RTState, gt_future_xyz: torch.Tensor | None = None,
                      where: float = 0.0) -> torch.Tensor:
        if self.subgoal_type == "latent":
            # Self-supervised parity loss: the future head should not collapse — supervise
            # it toward the current state tokens (a learnable residual delta carries motion).
            pred = self.imagine_tokens(state, self.horizon)
            return F.mse_loss(pred, state.memory.detach())
        # tracks: supervise predicted future xyz against full-clip GT (free, offline; spec §4).
        pred = self.imagine_tokens(state, self.horizon)
        if gt_future_xyz is None:
            return pred.sum() * 0.0          # no GT in this batch -> no-op (keeps graph valid)
        return F.l1_loss(pred, gt_future_xyz)
