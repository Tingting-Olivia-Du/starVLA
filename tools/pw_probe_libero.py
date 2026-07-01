# [LangPointWorld] PW-PROBE gate driver (C0, spec §4): frozen PointWorld on LIBERO demo
# actions -> imagined vs GT future flow in the EE frame -> corr/L2 verdict + per-episode npz.
# The single cheapest falsification of the LIBERO domain gap before any training.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")

from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher
from starVLA.model.modules.langpw.ee_frame import flow_to_ee_frame, assert_same_frame
from starVLA.model.modules.langpw import probe_metrics as M

T_MODEL = 11  # CONTEXT_HORIZON(1)+PRED_HORIZON(10); model predicts frames 1..10


def gt_ee_future_flow(hdf5, demo, t0, horizon):
    """GT key-point motion proxy = EE-position displacement over the horizon, world frame.
    The EE/gripper trajectory is the reliable arm-dim signal (same instrument as the FCDecoder
    under-drive probe); full-scene GT would need depth-tracked points."""
    with h5py.File(hdf5, "r") as f:
        ee = np.asarray(f["data"][demo]["obs"]["ee_pos"][:], np.float64)
    T = ee.shape[0]
    return np.stack([ee[min(t0 + h, T - 1)] - ee[t0] for h in range(1, horizon)], 0)  # [H-1,3]


def imagined_gripper_flow(out, data_dict, ee_pos, mode="motion"):
    """Reduce imagined full-scene flow to a per-step [H-1,3] world-frame displacement.

    mode='motion' (default): average the predicted displacement over the K points the MODEL
    predicts move most (highest total horizon displacement). This measures "does the model's
    predicted motion track the GT arm/object motion", which is the right question — selecting by
    proximity to EE at t=0 instead picks static background (the initial gate's methodology bug).
    mode='near_ee': legacy proximity selection (kept for comparison)."""
    scene_flows = out["scene_flows"]                      # [T,Ns,3] absolute positions
    coord0 = out["scene_coord0"]                          # [Ns,3]
    total_disp = torch.norm(scene_flows[-1] - scene_flows[0], dim=-1)  # [Ns]
    if mode == "near_ee":
        d0 = torch.norm(coord0 - torch.as_tensor(ee_pos, dtype=coord0.dtype), dim=-1)
        kp = torch.topk(-d0, k=min(64, d0.numel())).indices
    else:  # motion
        kp = torch.topk(total_disp, k=min(64, total_disp.numel())).indices
    disp = scene_flows[1:, kp, :] - scene_flows[0:1, kp, :].expand(scene_flows.shape[0] - 1, -1, -1)
    return disp.mean(dim=1)                               # [T-1,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libero-glob", required=True)
    ap.add_argument("--ptv3-size", default="large")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_probe")
    ap.add_argument("--corr-thresh", type=float, default=0.7)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(args.libero_glob))[: args.episodes]
    print(f"[probe] {len(files)} episodes; loading VGGT + teacher ...", flush=True)
    geo = VGGTGeometryProvider(device="cuda")
    builder = LiberoDataDictBuilder(domain="droid", device="cuda")
    teacher = PointWorldTeacher(args.ckpt, ptv3_size=args.ptv3_size, domain="droid", device="cuda")

    all_pred, all_gt = [], []
    for fi, hdf5 in enumerate(files):
        demo, t0 = "demo_0", 0
        try:
            dp = geo.depth_provider(hdf5, demo)
            K, E = geo.intrinsics_extrinsics(hdf5, demo, t0)
            s = build_pw_sample(hdf5, demo, t0, depth_provider=dp, domain="droid", intrinsics=K, extrinsics=E)
            with h5py.File(hdf5, "r") as f:
                obs = f["data"][demo]["obs"]
                joints = np.asarray(obs["joint_states"][:], np.float64)
                grip = np.asarray(obs["gripper_states"][:], np.float64)
                ee_pos = np.asarray(obs["ee_pos"][t0], np.float64)
                ee_ori = np.asarray(obs["ee_ori"][t0], np.float64)
            dd = builder.build(s, joints, grip, horizon=T_MODEL)
            out = teacher.imagine_from_datadict(dd)
            ee_quat = torch.as_tensor(s["ee_quat"], dtype=torch.float32)
            pred_world = imagined_gripper_flow(out, dd, ee_pos)                 # [T-1,3]
            gt_world = torch.as_tensor(gt_ee_future_flow(hdf5, demo, t0, T_MODEL), dtype=torch.float32)  # [T-1,3]
            pred_ee = flow_to_ee_frame(pred_world, ee_quat)
            gt_ee = flow_to_ee_frame(gt_world, ee_quat)
            assert_same_frame(t0, t0)
            all_pred.append(pred_ee.unsqueeze(0))
            all_gt.append(gt_ee.unsqueeze(0))
            np.savez(os.path.join(args.out_dir, f"ep{fi}.npz"),
                     pred_flow_ee=pred_ee.numpy(), gt_flow_ee=gt_ee.numpy(),
                     ee_pos=ee_pos, ee_quat=s["ee_quat"], hdf5=hdf5)
            print(f"[probe] ep{fi} ok: {os.path.basename(hdf5)}", flush=True)
        except Exception as e:
            import traceback
            print(f"[probe] ep{fi} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if not all_pred:
        print("[probe] no episodes succeeded — cannot compute verdict.", flush=True)
        return
    pred = torch.cat(all_pred, 0)   # [E,H-1,3]
    gt = torch.cat(all_gt, 0)
    mask = torch.ones(pred.shape[0], dtype=torch.bool)
    corr = M.keypoint_corr(pred, gt, mask)
    l2 = M.filtered_l2(pred, gt, mask)
    verdict = M.gate_verdict(corr["corr_mean"], args.corr_thresh)
    result = {**corr, **l2, "gate_pass": verdict, "n_episodes": int(pred.shape[0]),
              "ckpt": args.ckpt, "corr_thresh": args.corr_thresh}
    with open(os.path.join(args.out_dir, "probe_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    if verdict:
        print("\n[LangPointWorld] GATE PASSED — proceed to V0/V1.", flush=True)
    else:
        print("\n[LangPointWorld] GATE FAILED (spec §4 branch): retry large-droid+behavior ckpt "
              "-> short LIBERO finetune -> else STOP. Do NOT train on rejected labels.", flush=True)


if __name__ == "__main__":
    main()
