# [LangPointWorld] Batch check: for each LIBERO episode, does the finetuned PW predict the motion on
# the RIGHT (task) object, or does it put the movers on a WRONG object? PW is language-free, so it may
# confuse nearby objects. For each sim npz: identify the true task object (max GT displacement), run PW
# (official preprocessing + Panda robot_flows), find PW's predicted-mover points, and count how many
# land near each object. Report per-episode: task, correct-object mover count vs the top wrong object.
import argparse, glob, os, sys
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
FT = "/workspace/tingting/.tmp/s0_train/dummy-qbyz7nd4/model-last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="", help="comma list of sim npz basenames (no dir); default=one per task")
    ap.add_argument("--ckpt", default=FT)
    ap.add_argument("--move-thresh", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=0.10)
    args = ap.parse_args()

    if args.sims:
        npzs = [os.path.join("/workspace/tingting/.tmp/s0_sim", s if s.endswith(".npz") else s + ".npz")
                for s in args.sims.split(",")]
    else:
        # one demo per task (demo_0) across all 10 libero_object tasks
        tasks = sorted({os.path.basename(f).rsplit("_demo_", 1)[0]
                        for f in glob.glob("/workspace/tingting/.tmp/s0_sim/*_demo_*.npz")})
        npzs = [f"/workspace/tingting/.tmp/s0_sim/{tk}_demo_0.npz" for tk in tasks]
    npzs = [p for p in npzs if os.path.exists(p)]

    builder = LiberoDataDictBuilder(domain="droid", device="cuda", robot_urdf="panda", max_scene_points=8192)
    teacher = PointWorldTeacher(args.ckpt, ptv3_size="large", domain="droid", device="cuda")
    print(f"\n{'task':22s} {'demo':>4s} {'task_obj':22s} {'GTdisp':>6s} {'right#':>6s} {'wrong_obj':22s} {'wrong#':>6s} {'verdict':>8s}", flush=True)
    print("-" * 108, flush=True)
    n_right = n_total = 0
    for npz in npzs:
        base = os.path.basename(npz)[:-4]
        demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
        sr = np.load(npz, allow_pickle=True)
        names = list(sr["obj_names"]); op = sr["obj_poses"]
        objs = {n: op[0][names.index(n)][:3, 3] for n in names if n.endswith("_main")}
        disps = {n: np.linalg.norm(op[-1][names.index(n)][:3, 3] - op[0][names.index(n)][:3, 3]) for n in objs}
        task_obj = max(disps, key=disps.get); gt = disps[task_obj]

        H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
        with h5py.File(H, "r") as f:
            o = f["data"][f"demo_{demo}"]["obs"]
            j = np.asarray(o["joint_states"][:], np.float64); g = np.asarray(o["gripper_states"][:], np.float64)
        dd = builder.build_from_sim(sr, j, g, horizon=11, cams=("agentview",), frame="world", official_preprocess=True)
        pred = teacher.imagine_from_datadict(dd)["scene_flows"].numpy()
        shift = np.asarray(dd.get("__shift_amount__", np.zeros(3)), np.float32)
        disp = np.linalg.norm(pred[-1] - pred[0], axis=-1)
        mv = pred[0][disp > args.move_thresh]
        # count movers near each object (exclude basket)
        cnt = {n: int((np.linalg.norm(mv - (c + shift), axis=-1) < args.radius).sum())
               for n, c in objs.items() if "basket" not in n}
        right = cnt.get(task_obj, 0)
        wrong_obj = max((n for n in cnt if n != task_obj), key=lambda n: cnt[n], default="-")
        wrong = cnt.get(wrong_obj, 0)
        ok = right >= wrong and right > 0
        n_right += int(ok); n_total += 1
        print(f"{task:22s} {demo:>4s} {task_obj[:22]:22s} {gt:6.2f} {right:6d} {wrong_obj[:22]:22s} {wrong:6d} {'RIGHT' if ok else 'WRONG':>8s}", flush=True)
    print("-" * 108, flush=True)
    print(f"[wrong-obj] {n_right}/{n_total} episodes put MOST movers on the correct task object "
          f"({100*n_right/max(n_total,1):.0f}%). WRONG = PW confused a nearby object.", flush=True)


if __name__ == "__main__":
    main()
