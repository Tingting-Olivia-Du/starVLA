# [LangPointWorld] Compare RAW vs OFFICIAL-preprocessed scene cloud in one viser. RAW = our
# build_from_sim (bounds + random subsample, robot points now removed via seg). OFFICIAL = + the
# official PointWorld test-mode preprocessing (center_shift -> voxel 1.5cm grid_sample -> enforce
# max 12000 -> center_shift -> normalize_colors), matching what PW was trained on. Dropdown switches;
# also shows robot_flows (magenta) so you can confirm the robot is SEPARATE from the scene cloud now.
import argparse, os, socket, sys, time
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"


def free_port(p):
    while True:
        s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
        if not busy:
            return p
        p += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="/workspace/tingting/.tmp/s0_sim/alphabet_soup_demo_0.npz")
    ap.add_argument("--port", type=int, default=8094)
    args = ap.parse_args()

    from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
    base = os.path.basename(args.sim)[:-4]
    demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
    z = np.load(args.sim, allow_pickle=True)
    with h5py.File(H, "r") as f:
        o = f["data"][f"demo_{demo}"]["obs"]
        j = np.asarray(o["joint_states"][:], np.float64); g = np.asarray(o["gripper_states"][:], np.float64)

    b = LiberoDataDictBuilder(domain="droid", device="cuda", max_scene_points=30000)
    variants = {}
    specs = [("RAW (ours)", dict(official_preprocess=False)),
             ("OFFICIAL (voxel+center+12k)", dict(official_preprocess=True)),
             ("OFFICIAL + object-foveated", dict(official_preprocess=True, official_importance=True))]
    for tag, kw in specs:
        dd = b.build_from_sim(z, j, g, horizon=11, cams=("agentview", "robot0_eye_in_hand"),
                              frame="world", **kw)
        sf = dd["scene_flows"][0].astype(np.float32)
        col = dd["_scene_colors_u8"][:sf.shape[0]].astype(np.uint8)
        rf = dd["robot_flows"][0].astype(np.float32)
        variants[tag] = (sf, col, rf)
        print(f"[pre] {tag:32s} scene_Ns={sf.shape[0]:6d} robot_Nr={rf.shape[0]}", flush=True)

    import viser
    p = free_port(args.port)
    server = viser.ViserServer(port=p); server.scene.set_up_direction("+z")
    labels = list(variants.keys())
    sel = server.gui.add_dropdown("cloud", labels, labels[-1])
    gui_rob = server.gui.add_checkbox("robot_flows (magenta, SEPARATE from scene)", True)

    def render():
        sf, col, rf = variants[sel.value]
        server.scene.add_point_cloud("scene", sf, colors=col, point_size=0.005)
        if gui_rob.value:
            server.scene.add_point_cloud("robot", rf, colors=(255, 0, 255), point_size=0.007)
        else:
            server.scene.remove_by_name("robot")
    sel.on_update(lambda _: render()); gui_rob.on_update(lambda _: render()); render()
    print(f"[pre] viser READY  http://localhost:{p}   (ACTUAL PORT {p}) | dropdown=RAW/OFFICIAL, "
          f"magenta=robot (should be separate from scene now)", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
