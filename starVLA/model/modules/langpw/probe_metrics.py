# [LangPointWorld] PW-PROBE gate metrics: per-dim corr + filtered L2, EE frame (spec §4).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def keypoint_corr(pred_flow_ee, gt_flow_ee, mask):
    """Per-dim Pearson correlation of imagined vs GT displacement over masked key points."""
    p = pred_flow_ee[mask]  # [M,H,3]
    g = gt_flow_ee[mask]
    cx, cy, cz = (_pearson(p[..., i], g[..., i]) for i in range(3))
    return {"corr_x": cx, "corr_y": cy, "corr_z": cz, "corr_mean": (cx + cy + cz) / 3.0}


def filtered_l2(pred_flow_ee, gt_flow_ee, mask):
    """Endpoint L2 error over masked points (PointWorld filtered_l2_moved style)."""
    p = pred_flow_ee[mask]
    g = gt_flow_ee[mask]
    l2 = torch.norm(p - g, dim=-1).reshape(-1)
    return {
        "l2_mean": float(l2.mean()),
        "l2_p50": float(l2.median()),
        "l2_p90": float(l2.quantile(0.90)),
    }


def gate_verdict(corr_mean: float, corr_thresh: float = 0.7) -> bool:
    """PW-PROBE pass criterion (spec §4): key-point mean correlation exceeds threshold."""
    return bool(corr_mean > corr_thresh)
