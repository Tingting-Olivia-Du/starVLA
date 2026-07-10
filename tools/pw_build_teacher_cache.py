# [LangPointWorld] S1-a: build the teacher-flow distillation cache. For each LIBERO demo:
#   1. build_from_sim(frame="world") -> frozen finetuned PW imagine -> scene_flows[T,Ns,3] (world)
#   2. object/gripper masks by proximity to GT object t0 center + EE t0 pos (no seg plumbing needed)
#   3. sample N points (bias object/gripper/high-motion) via langpw.point_sampling.sample_points
#   4. teacher flow of those N points -> EE frame at t0 (langpw.ee_frame) -> F_teacher[N,Hf,3]
#   5. store weights w[N] + the student-input payload (agentview+wrist RGB, proprio) + frame_id
# One cache npz per demo. This is the distillation supervision the RGB-only student regresses to.
import argparse, glob, os, sys
import numpy as np
import h5py
import torch

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher
from starVLA.model.modules.langpw.point_sampling import sample_points
from starVLA.model.modules.langpw.ee_frame import to_ee_frame, flow_to_ee_frame

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11


def eul2q(e):
    cx, cy, cz = np.cos(np.asarray(e) * 0.5); sx, sy, sz = np.sin(np.asarray(e) * 0.5)
    return np.array([sx*cy*cz - cx*sy*sz, cx*sy*cz + sx*cy*sz,
                     cx*cy*sz - sx*sy*cz, cx*cy*cz + sx*sy*sz], np.float32)  # [x,y,z,w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sim-glob", default="/workspace/tingting/.tmp/s0_sim/*_demo_*.npz")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/s1_cache")
    ap.add_argument("--n-points", type=int, default=256)
    ap.add_argument("--obj-radius", type=float, default=0.09)
    ap.add_argument("--grip-radius", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    npzs = sorted(glob.glob(args.sim_glob))
    if args.limit > 0:
        npzs = npzs[:args.limit]
    builder = LiberoDataDictBuilder(domain="droid", device="cuda")
    teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")
    print(f"[cache] {len(npzs)} demos -> {args.out_dir} (N={args.n_points})", flush=True)

    ok = 0
    for npz in npzs:
        base = os.path.basename(npz)[:-4]
        demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
        out = os.path.join(args.out_dir, f"{task}__demo_{demo}.npz")
        if os.path.exists(out):
            ok += 1; continue
        try:
            sr = np.load(npz, allow_pickle=True)
            names = list(sr["obj_names"]); op = sr["obj_poses"]
            cand = [n for n in names if not n.startswith(("robot0", "gripper", "mount"))
                    and n not in ("floor", "table", "world")]
            disps = {n: np.linalg.norm(op[-1][names.index(n)][:3, 3] - op[0][names.index(n)][:3, 3]) for n in cand}
            obj = max(disps, key=disps.get); obj_ctr = op[0][names.index(obj)][:3, 3]

            H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
            with h5py.File(H, "r") as f:
                o = f["data"][f"demo_{demo}"]["obs"]
                joints = np.asarray(o["joint_states"][:], np.float64)
                grip = np.asarray(o["gripper_states"][:], np.float64)
                ee_pos = np.asarray(o["ee_pos"][:], np.float64)
                ee_ori = np.asarray(o["ee_ori"][:], np.float64)
            ti = sr["ti"]
            ee_pos_t0 = ee_pos[ti[0]].astype(np.float32)
            ee_quat_t0 = eul2q(ee_ori[ti[0]])

            dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=("agentview",), frame="world")
            pred_world = teacher.imagine_from_datadict(dd)["scene_flows"].numpy()   # [T,Ns,3] world
            T, Ns, _ = pred_world.shape
            coord0 = pred_world[0]                                                  # [Ns,3] world

            # proximity masks (world frame)
            obj_mask = torch.from_numpy(np.linalg.norm(coord0 - obj_ctr, axis=-1) < args.obj_radius)
            grip_mask = torch.from_numpy(np.linalg.norm(coord0 - ee_pos_t0, axis=-1) < args.grip_radius)
            target_mask = torch.zeros(Ns, dtype=torch.bool)   # no explicit target obj on this task set

            flow_world = torch.from_numpy(pred_world.astype(np.float32))            # [T,Ns,3] absolute pos
            # sample_points expects DISPLACEMENT flow (its `motion` = sum of per-frame norms). Pass the
            # delta-from-t0 so motion-ranking + the (1+motion) weight reflect true point displacement,
            # not distance-from-origin (which would make static far points look "high motion").
            delta_all = flow_world - flow_world[0:1]                                 # [T,Ns,3]
            samp = sample_points(torch.from_numpy(coord0.astype(np.float32)), delta_all,
                                 obj_mask, target_mask, grip_mask, n_points=args.n_points)
            idx = samp["indices"]                                                   # [N]
            w = samp["weight"].float().numpy()                                      # [N]

            # teacher flow of the sampled N points, DELTA vs t0, in EE frame at t0.
            # F_teacher[t] = flow_world[t] - flow_world[0], expressed in EE frame (rotation only).
            pts0 = flow_world[0][idx]                                               # [N,3] world
            delta_world = flow_world[:, idx, :] - flow_world[0:1, idx, :]           # [T,N,3]
            ee_quat_t = torch.from_numpy(ee_quat_t0)
            F_ee = flow_to_ee_frame(delta_world.reshape(-1, 3), ee_quat_t).reshape(T, len(idx), 3)  # [T,N,3]
            pts0_ee = to_ee_frame(pts0, torch.from_numpy(ee_pos_t0), ee_quat_t)     # [N,3]
            # store as [N, Hf, 3] with Hf = T-1 (future frames 1..T-1 relative to t0)
            F_teacher = F_ee[1:].permute(1, 0, 2).contiguous().numpy().astype(np.float32)  # [N,Hf,3]

            # student input payload (RGB-only deployable): both cam RGB at t0 + proprio
            av_rgb0 = sr["agentview_rgb"][0]                                        # [128,128,3] u8
            wr_rgb0 = sr["robot0_eye_in_hand_rgb"][0]
            proprio_t0 = np.concatenate([joints[ti[0]], grip[ti[0]]]).astype(np.float32)  # [9]

            np.savez(out,
                     F_teacher=F_teacher,                     # [N,Hf,3] EE frame, delta-from-t0
                     teacher_weight=w.astype(np.float32),     # [N]
                     teacher_points0_ee=pts0_ee.numpy().astype(np.float32),  # [N,3]
                     teacher_points0_world=pts0.numpy().astype(np.float32),  # [N,3] (for viz gate)
                     role=samp["role"].numpy(),               # [N] int role id
                     frame_id=np.int64(int(ti[0])),
                     ee_pos_t0=ee_pos_t0, ee_quat_t0=ee_quat_t0,
                     agentview_rgb=av_rgb0, wrist_rgb=wr_rgb0,
                     proprio=proprio_t0, task=task, demo=demo,
                     obj_name=obj, obj_traj_world=op[:, names.index(obj), :3, 3].astype(np.float32),
                     cam0_intrinsic=dd["cam0_intrinsic"].astype(np.float64),
                     cam0_extrinsic=dd["cam0_extrinsic"].astype(np.float64))
            ok += 1
            if ok % 20 == 0:
                fmag = np.linalg.norm(F_teacher[:, -1], axis=-1)
                print(f"[cache] {ok}/{len(npzs)} last={task} demo_{demo} obj={obj} N={len(idx)} "
                      f"objpts={int(obj_mask.sum())} F_last_mean={fmag.mean():.3f}", flush=True)
        except Exception as e:
            print(f"[cache] FAIL {base}: {type(e).__name__}: {e}", flush=True)
    print(f"[cache] DONE ok={ok}/{len(npzs)}", flush=True)


if __name__ == "__main__":
    main()
