# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/metrics.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch


def depth_absrel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred[mask]
    target = target[mask].clamp_min(1e-6)
    return ((pred - target).abs() / target).mean()


def depth_delta1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred[mask].clamp_min(1e-6)
    target = target[mask].clamp_min(1e-6)
    ratio = torch.maximum(pred / target, target / pred)
    return (ratio < 1.25).float().mean()


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(x, y)
    return dists.min(dim=-1).values.mean() + dists.min(dim=-2).values.mean()


def translation_error(pred_extrinsics: torch.Tensor, gt_extrinsics: torch.Tensor) -> torch.Tensor:
    return (pred_extrinsics[..., :3, 3] - gt_extrinsics[..., :3, 3]).norm(dim=-1).mean()


def _safe_stack_mean(values: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.stack(values).mean()


def farthest_point_indices(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return torch.arange(points.shape[0], device=points.device)
    selected = torch.empty(max_points, dtype=torch.long, device=points.device)
    min_dist = torch.full((points.shape[0],), float('inf'), device=points.device, dtype=points.dtype)
    farthest = torch.randint(0, points.shape[0], (1,), device=points.device).item()
    for idx in range(max_points):
        selected[idx] = farthest
        centroid = points[farthest : farthest + 1]
        dist = (points - centroid).pow(2).sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist).item()
    return selected


def umeyama_alignment(
    src: torch.Tensor,
    dst: torch.Tensor,
    with_scale: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[-1] != 3:
        raise ValueError(f'Expected matching Nx3 point sets, got {src.shape} and {dst.shape}.')
    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = dst_centered.transpose(0, 1) @ src_centered / max(src.shape[0], 1)
    u, singular_values, vh = torch.linalg.svd(covariance)
    correction = torch.ones(3, device=src.device, dtype=src.dtype)
    if torch.det(u @ vh) < 0:
        correction[-1] = -1.0
    rotation = u @ torch.diag(correction) @ vh
    if with_scale:
        variance = src_centered.pow(2).sum() / max(src.shape[0], 1)
        scale = (singular_values * correction).sum() / variance.clamp_min(eps)
    else:
        scale = torch.ones((), device=src.device, dtype=src.dtype)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(points: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return scale * (points @ rotation.transpose(0, 1)) + translation


def _point_cloud_metrics_from_points(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_idx = farthest_point_indices(target, max_points)
    pred = pred[sample_idx]
    target = target[sample_idx]
    scale, rotation, translation = umeyama_alignment(pred, target, with_scale=True)
    pred_aligned = apply_similarity(pred, scale, rotation, translation)
    dists = torch.cdist(pred_aligned, target)
    accuracy = dists.min(dim=-1).values.mean()
    completeness = dists.min(dim=-2).values.mean()
    chamfer = accuracy + completeness
    return accuracy, completeness, chamfer


def point_cloud_forecast_metrics(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
    max_points: int = 20000,
) -> dict[str, torch.Tensor]:
    accuracy_values = []
    completeness_values = []
    chamfer_values = []
    device = pred_points.device
    dtype = pred_points.dtype

    for batch_idx in range(pred_points.shape[0]):
        for frame_idx in range(pred_points.shape[1]):
            valid = mask[batch_idx, frame_idx].bool()
            if valid.sum() < 8:
                continue
            pred = pred_points[batch_idx, frame_idx][valid]
            target = target_points[batch_idx, frame_idx][valid]
            accuracy, completeness, chamfer = _point_cloud_metrics_from_points(pred, target, max_points=max_points)
            accuracy_values.append(accuracy)
            completeness_values.append(completeness)
            chamfer_values.append(chamfer)

    return {
        'metric_point_accuracy': _safe_stack_mean(accuracy_values, device, dtype),
        'metric_point_completeness': _safe_stack_mean(completeness_values, device, dtype),
        'metric_point_chamfer': _safe_stack_mean(chamfer_values, device, dtype),
    }


def aggregated_point_cloud_forecast_metrics(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
    max_points: int = 20000,
) -> dict[str, torch.Tensor]:
    accuracy_values = []
    completeness_values = []
    chamfer_values = []
    device = pred_points.device
    dtype = pred_points.dtype

    for batch_idx in range(pred_points.shape[0]):
        valid = mask[batch_idx].bool()
        if valid.sum() < 8:
            continue
        pred = pred_points[batch_idx][valid]
        target = target_points[batch_idx][valid]
        accuracy, completeness, chamfer = _point_cloud_metrics_from_points(pred, target, max_points=max_points)
        accuracy_values.append(accuracy)
        completeness_values.append(completeness)
        chamfer_values.append(chamfer)

    return {
        'metric_point_accuracy': _safe_stack_mean(accuracy_values, device, dtype),
        'metric_point_completeness': _safe_stack_mean(completeness_values, device, dtype),
        'metric_point_chamfer': _safe_stack_mean(chamfer_values, device, dtype),
    }


def _extrinsics_to_c2w(extrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_cw = extrinsics[..., :3, :3]
    translation_cw = extrinsics[..., :3, 3]
    rotation_wc = rotation_cw.transpose(-1, -2)
    camera_center = -(rotation_wc @ translation_cw.unsqueeze(-1)).squeeze(-1)
    return rotation_wc, camera_center


def _c2w_to_extrinsics(rotation_wc: torch.Tensor, camera_center: torch.Tensor) -> torch.Tensor:
    rotation_cw = rotation_wc.transpose(-1, -2)
    translation_cw = -(rotation_cw @ camera_center.unsqueeze(-1)).squeeze(-1)
    return torch.cat([rotation_cw, translation_cw.unsqueeze(-1)], dim=-1)


def _rotation_geodesic_deg(rotation_a: torch.Tensor, rotation_b: torch.Tensor) -> torch.Tensor:
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def align_camera_trajectory(pred_extrinsics: torch.Tensor, gt_extrinsics: torch.Tensor) -> torch.Tensor:
    pred_rotation_wc, pred_center = _extrinsics_to_c2w(pred_extrinsics)
    gt_rotation_wc, gt_center = _extrinsics_to_c2w(gt_extrinsics)
    scale, rotation_align, translation_align = umeyama_alignment(pred_center, gt_center, with_scale=True)
    pred_center_aligned = apply_similarity(pred_center, scale, rotation_align, translation_align)
    pred_rotation_aligned = rotation_align.unsqueeze(0) @ pred_rotation_wc
    return _c2w_to_extrinsics(pred_rotation_aligned, pred_center_aligned)


def camera_trajectory_metrics(pred_extrinsics: torch.Tensor, gt_extrinsics: torch.Tensor) -> dict[str, torch.Tensor]:
    ate_values = []
    rte_values = []
    rre_values = []
    device = pred_extrinsics.device
    dtype = pred_extrinsics.dtype

    for batch_idx in range(pred_extrinsics.shape[0]):
        pred_aligned = align_camera_trajectory(pred_extrinsics[batch_idx], gt_extrinsics[batch_idx])
        pred_rotation_wc, pred_center = _extrinsics_to_c2w(pred_aligned)
        gt_rotation_wc, gt_center = _extrinsics_to_c2w(gt_extrinsics[batch_idx])
        ate_values.append(torch.sqrt(((pred_center - gt_center).pow(2).sum(dim=-1)).mean()))

        if pred_center.shape[0] > 1:
            pred_rot_prev = pred_rotation_wc[:-1]
            pred_rot_next = pred_rotation_wc[1:]
            gt_rot_prev = gt_rotation_wc[:-1]
            gt_rot_next = gt_rotation_wc[1:]
            pred_center_prev = pred_center[:-1]
            pred_center_next = pred_center[1:]
            gt_center_prev = gt_center[:-1]
            gt_center_next = gt_center[1:]

            pred_rel_rot = pred_rot_prev.transpose(-1, -2) @ pred_rot_next
            gt_rel_rot = gt_rot_prev.transpose(-1, -2) @ gt_rot_next
            pred_rel_trans = (pred_rot_prev.transpose(-1, -2) @ (pred_center_next - pred_center_prev).unsqueeze(-1)).squeeze(-1)
            gt_rel_trans = (gt_rot_prev.transpose(-1, -2) @ (gt_center_next - gt_center_prev).unsqueeze(-1)).squeeze(-1)

            rte_values.append((pred_rel_trans - gt_rel_trans).norm(dim=-1).mean())
            rre_values.append(_rotation_geodesic_deg(pred_rel_rot, gt_rel_rot).mean())

    return {
        'metric_camera_ate': _safe_stack_mean(ate_values, device, dtype),
        'metric_camera_rte': _safe_stack_mean(rte_values, device, dtype),
        'metric_camera_rre': _safe_stack_mean(rre_values, device, dtype),
    }
