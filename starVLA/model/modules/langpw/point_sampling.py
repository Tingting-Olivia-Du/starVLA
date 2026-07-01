# [LangPointWorld] Hybrid sparse point selection (C3, spec §5.3): allocate N points across
# object / target / gripper / background roles from LIBERO sim seg masks, filling shortfalls
# from teacher-flow motion magnitude then uniform; weight up-scales moving/object/gripper.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch

_ROLE_WEIGHT = {0: 1.0, 1: 0.7, 2: 1.0, 3: 0.3}  # obj, target, gripper, background


def _pick(pool_idx, k, motion, used):
    """Pick k indices from pool_idx (a 1D LongTensor), preferring high motion, avoiding `used`."""
    avail = [int(i) for i in pool_idx.tolist() if int(i) not in used]
    if len(avail) == 0:
        return []
    avail_t = torch.tensor(avail)
    if len(avail) <= k:
        chosen = avail_t
    else:
        m = motion[avail_t]
        chosen = avail_t[torch.topk(m, k).indices]
    for i in chosen.tolist():
        used.add(int(i))
    return chosen.tolist()


def sample_points(scene_coord0, scene_flow, object_mask, target_mask, gripper_mask,
                  n_points=256, alloc=(96, 48, 48, 64)):
    """scene_coord0:[Ns,3]; scene_flow:[Hf,Ns,3]; masks:[Ns] bool.
    Returns {indices:[N], weight:[N], role:[N]} (role 0 obj,1 target,2 gripper,3 bg)."""
    Ns = scene_coord0.shape[0]
    motion = scene_flow.norm(dim=-1).sum(dim=0)  # [Ns] total motion magnitude per point
    all_idx = torch.arange(Ns)
    masks = [object_mask, target_mask, gripper_mask, torch.ones(Ns, dtype=torch.bool)]
    used = set()
    chosen_idx, chosen_role = [], []
    for role, (mask, k) in enumerate(zip(masks, alloc)):
        pool = all_idx[mask.bool()] if mask.any() else all_idx
        picked = _pick(pool, k, motion, used)
        # shortfall: fill from global motion-ranked, then uniform
        if len(picked) < k:
            picked += _pick(all_idx, k - len(picked), motion, used)
        chosen_idx += picked
        chosen_role += [role] * len(picked)
    # pad to exactly n_points if still short (tiny scenes)
    while len(chosen_idx) < n_points:
        picked = _pick(all_idx, n_points - len(chosen_idx), motion, used)
        if not picked:
            # scene smaller than n_points: allow repeats to keep a fixed N
            picked = [int(all_idx[len(chosen_idx) % Ns])]
        chosen_idx += picked
        chosen_role += [3] * len(picked)
    idx = torch.tensor(chosen_idx[:n_points])
    role = torch.tensor(chosen_role[:n_points])
    base_w = torch.tensor([_ROLE_WEIGHT[int(r)] for r in role])
    motion_factor = 1.0 + motion[idx]
    weight = base_w * motion_factor
    return {"indices": idx, "weight": weight, "role": role}
