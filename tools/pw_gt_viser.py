# [LangPointWorld] Interactive viser for the ACCURATE GT scene flow (from the packed droid HDF5).
# Scene point cloud in true colors; GT-moving points highlighted; frame slider + toggles. Lets you
# 3D-inspect that the GT moving points hug the manipulated object and follow it into the basket.
import argparse, time
import numpy as np
import h5py
import viser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="packed droid-format HDF5 (fixed-flip GT)")
    ap.add_argument("--clip", default="0:11")
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--move-thresh", type=float, default=0.05)
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        cam = f[args.clip]["camera_0"]
        sf = np.asarray(cam["scene_flows"]).astype(np.float32)     # [T,Ns,3] world (GT)
        colors = np.asarray(cam["scene_colors"])                   # [T,Ns,3] u8
    T, Ns, _ = sf.shape
    col0 = colors[0] if colors.ndim == 3 else colors               # [Ns,3]
    disp_total = np.linalg.norm(sf[-1] - sf[0], axis=-1)
    moving = disp_total > args.move_thresh

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    gui_t = server.gui.add_slider("frame", 0, T - 1, 1, T - 1)
    gui_hi = server.gui.add_checkbox("highlight GT-moving points (red)", True)
    gui_only = server.gui.add_checkbox("show ONLY moving points", False)

    def render():
        t = int(gui_t.value)
        pts = sf[t]
        cols = col0.copy()
        if gui_hi.value:
            cols[moving] = np.array([255, 0, 0], np.uint8)         # moving -> red
        if gui_only.value:
            server.scene.add_point_cloud("scene", pts[moving].astype(np.float32),
                                         colors=cols[moving].astype(np.uint8), point_size=0.006)
        else:
            server.scene.add_point_cloud("scene", pts.astype(np.float32),
                                         colors=cols.astype(np.uint8), point_size=0.004)

    for g in (gui_t, gui_hi, gui_only):
        g.on_update(lambda _: render())
    render()
    print(f"[gt-viser] READY http://localhost:{args.port} | {Ns} pts, {int(moving.sum())} GT-moving "
          f"(>{args.move_thresh}m), max disp {disp_total.max():.3f} | red=moving GT", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
