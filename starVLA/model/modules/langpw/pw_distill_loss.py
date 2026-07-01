# [LangPointWorld] Distillation loss (C5, spec §5.5): weighted smooth-L1 between student
# and cached teacher point-flow, both in the same-timestep EE frame (same-frame asserted).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch
import torch.nn.functional as F
from starVLA.model.modules.langpw.ee_frame import assert_same_frame


def pw_distill_loss(flow_student, flow_teacher, weight, frame_id_student, frame_id_teacher):
    """flow_*: [B,N,Hf,3]; weight: [B,N]; frame ids: ints (same-timestep EE frame guard).

    Returns a scalar weighted smooth-L1. Weight up-weights moving/object/gripper points and
    down-weights static/low-confidence points (from the cache's confidence field, spec §5.5).
    """
    assert_same_frame(frame_id_student, frame_id_teacher)
    per_elem = F.smooth_l1_loss(flow_student, flow_teacher, reduction="none")  # [B,N,Hf,3]
    per_point = per_elem.mean(dim=(-1, -2))                                     # [B,N]
    w = weight.to(per_point.dtype)
    denom = w.sum().clamp_min(1e-6)
    return (per_point * w).sum() / denom
