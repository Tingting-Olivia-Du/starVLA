# [LangPointWorld] EE-centric ("robo-centric") frame transform — cross-dataset
# unification frame for teacher/GT/student point-flow (spec §4).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # q = (x,y,z,w); conjugate = (-x,-y,-z,w). For unit quats, conjugate == inverse.
    return torch.stack([-q[0], -q[1], -q[2], q[3]])


def _rotate_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Rotate vectors v[...,3] by unit quaternion q=(x,y,z,w). Rodrigues/Hamilton form.
    x, y, z, w = q[0], q[1], q[2], q[3]
    qv = torch.stack([x, y, z]).to(v.dtype)
    uv = torch.cross(qv.expand_as(v), v, dim=-1)
    uuv = torch.cross(qv.expand_as(v), uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def flow_to_ee_frame(flow_world: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
    """A displacement rotates only (world->EE uses the inverse EE rotation)."""
    ee_quat = ee_quat.to(flow_world.dtype)
    return _rotate_by_quat(flow_world, _quat_conjugate(ee_quat))


def to_ee_frame(points_world: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
    """Express world-frame points in the EE frame: translate then inverse-rotate."""
    ee_pos = ee_pos.to(points_world.dtype)
    ee_quat = ee_quat.to(points_world.dtype)
    return _rotate_by_quat(points_world - ee_pos, _quat_conjugate(ee_quat))


def assert_same_frame(frame_id_a: int, frame_id_b: int) -> None:
    """Guard: EE origin moves per timestep; only compare arrays from the same frame."""
    if int(frame_id_a) != int(frame_id_b):
        raise ValueError(
            f"[LangPointWorld] EE-frame mismatch: {frame_id_a} vs {frame_id_b}. "
            "Teacher/GT/student flow must be compared in the same timestep's EE frame (spec §4)."
        )
