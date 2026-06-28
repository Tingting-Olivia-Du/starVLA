# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/losses.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metrics import (
    aggregated_point_cloud_forecast_metrics,
    camera_trajectory_metrics,
    depth_absrel,
    depth_delta1,
    point_cloud_forecast_metrics,
    translation_error,
)


def _to_long_tensor(value, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor[None]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.long()


@dataclass(eq=False)
class WorldModelLoss(torch.nn.Module):
    latent_weight: float = 1.0
    decode_depth_weight: float = 0.0
    decode_point_weight: float = 0.0
    decode_camera_weight: float = 0.0

    def __init__(
        self,
        latent_weight: float = 1.0,
        decode_depth_weight: float = 0.0,
        decode_point_weight: float = 0.0,
        decode_camera_weight: float = 0.0,
    ):
        super().__init__()
        self.latent_weight = latent_weight
        self.decode_depth_weight = decode_depth_weight
        self.decode_point_weight = decode_point_weight
        self.decode_camera_weight = decode_camera_weight

    def _slice_prediction_frames(self, predictions, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            return tensor
        start = int(predictions.get('decoded_frame_start', 0))
        count = int(predictions.get('decoded_frame_count', max(tensor.shape[1] - start, 0)))
        return tensor[:, start : start + count]

    def _slice_batch_frames(self, predictions, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            return tensor
        start = int(predictions.get('batch_frame_start', 0))
        count = int(predictions.get('decoded_frame_count', max(tensor.shape[1] - start, 0)))
        return tensor[:, start : start + count]

    def _match_frames_by_id(self, predictions, batch, pred_tensor: torch.Tensor, batch_tensor: torch.Tensor, key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        decoded_ids = _to_long_tensor(predictions.get('decoded_frame_ids'), pred_tensor.device)
        batch_ids = _to_long_tensor(batch.get('ids'), pred_tensor.device)
        metric_ids = _to_long_tensor(batch.get(key), pred_tensor.device)
        if decoded_ids is None or batch_ids is None or metric_ids is None:
            return None

        pred_selected = []
        batch_selected = []
        for batch_idx in range(metric_ids.shape[0]):
            pred_idx = []
            batch_idx_sel = []
            for frame_id in metric_ids[batch_idx].tolist():
                pred_matches = (decoded_ids[batch_idx] == frame_id).nonzero(as_tuple=False)
                batch_matches = (batch_ids[batch_idx] == frame_id).nonzero(as_tuple=False)
                if pred_matches.numel() == 0 or batch_matches.numel() == 0:
                    continue
                pred_idx.append(int(pred_matches[0, 0]))
                batch_idx_sel.append(int(batch_matches[0, 0]))
            if not pred_idx:
                continue
            pred_selected.append(pred_tensor[batch_idx, pred_idx])
            batch_selected.append(batch_tensor[batch_idx, batch_idx_sel])

        if not pred_selected:
            return None
        return torch.stack(pred_selected), torch.stack(batch_selected)

    def _select_metric_frames(self, predictions, batch, pred_tensor: torch.Tensor, batch_tensor: torch.Tensor, key: str = 'metric_frame_indices') -> tuple[torch.Tensor, torch.Tensor]:
        matched = self._match_frames_by_id(predictions, batch, pred_tensor, batch_tensor, key)
        if matched is not None:
            return matched
        return self._slice_prediction_frames(predictions, pred_tensor), self._slice_batch_frames(predictions, batch_tensor)

    def _latent_loss(self, predictions) -> torch.Tensor:
        pred_state = predictions['pred_state_tokens']
        target_state = predictions['target_state_tokens']
        if pred_state.shape[1] != target_state.shape[1]:
            target_state = target_state[:, -pred_state.shape[1] :]

        stage = predictions.get('stage', 'eval')
        tau = predictions.get('tau')
        if stage == 'stage1' and tau is not None:
            while tau.ndim < pred_state.ndim:
                tau = tau.unsqueeze(-1)
            weight = 1.0 / (1.0 - tau).clamp_min(1e-2).pow(2)
            return ((pred_state - target_state).pow(2) * weight).mean()
        if stage == 'stage2':
            return F.mse_loss(pred_state, target_state)
        return F.mse_loss(pred_state, target_state)

    def forward(self, predictions, batch) -> torch.Tensor:
        if 'pred_state_tokens' not in predictions or 'target_state_tokens' not in predictions:
            raise KeyError('Expected `pred_state_tokens` and `target_state_tokens` in predictions.')

        loss_latent = self._latent_loss(predictions)
        total = self.latent_weight * loss_latent
        out = {
            'loss_objective': total,
            'loss_latent': loss_latent,
        }

        if 'decoded_depth' in predictions and 'depths' in batch and 'point_masks' in batch:
            pred_depth, gt_depth = self._select_metric_frames(predictions, batch, predictions['decoded_depth'], batch['depths'])
            _, mask = self._select_metric_frames(predictions, batch, predictions['decoded_depth'], batch['point_masks'])
            pred_depth = pred_depth.squeeze(-1)
            mask = mask.bool()
            if mask.any():
                out['metric_depth_absrel'] = depth_absrel(pred_depth, gt_depth, mask)
                out['metric_depth_delta1'] = depth_delta1(pred_depth, gt_depth, mask)
                if self.decode_depth_weight > 0:
                    decode_depth = F.l1_loss(pred_depth[mask], gt_depth[mask])
                    total = total + self.decode_depth_weight * decode_depth
                    out['loss_decode_depth'] = decode_depth

        if 'decoded_world_points' in predictions and 'world_points' in batch and 'point_masks' in batch:
            aggregate_key = 'aggregate_world_points_frame_ids'
            if batch.get(aggregate_key) is not None:
                pred_points, gt_points = self._select_metric_frames(predictions, batch, predictions['decoded_world_points'], batch['world_points'], key=aggregate_key)
                _, mask = self._select_metric_frames(predictions, batch, predictions['decoded_world_points'], batch['point_masks'], key=aggregate_key)
                point_metrics = aggregated_point_cloud_forecast_metrics(pred_points, gt_points, mask.bool())
            else:
                pred_points, gt_points = self._select_metric_frames(predictions, batch, predictions['decoded_world_points'], batch['world_points'])
                _, mask = self._select_metric_frames(predictions, batch, predictions['decoded_world_points'], batch['point_masks'])
                point_metrics = point_cloud_forecast_metrics(pred_points, gt_points, mask.bool())
            out.update(point_metrics)
            mask = mask.bool()
            if self.decode_point_weight > 0 and mask.any():
                point_loss = F.l1_loss(pred_points[mask], gt_points[mask])
                total = total + self.decode_point_weight * point_loss
                out['loss_decode_point'] = point_loss

        if 'decoded_extrinsics' in predictions and 'extrinsics' in batch:
            pred_extrinsics, gt_extrinsics = self._select_metric_frames(predictions, batch, predictions['decoded_extrinsics'], batch['extrinsics'])
            camera_metrics = camera_trajectory_metrics(pred_extrinsics, gt_extrinsics)
            out.update(camera_metrics)
            if self.decode_camera_weight > 0:
                camera_loss = translation_error(pred_extrinsics, gt_extrinsics)
                total = total + self.decode_camera_weight * camera_loss
                out['loss_decode_camera'] = camera_loss

        out['objective'] = total
        return out
