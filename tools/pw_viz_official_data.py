# [LangPointWorld] Drive the OFFICIAL PredictionVisualizer on OFFICIAL PointWorld data, headless.
# Replicates evaluation/tester.py:_visualize_eval_samples but WITHOUT the interactive-TTY prompt:
# it visualizes the first sample and HOLDS the viser server open (the eval loop otherwise blocks
# on input() and closes live_session in finally). Lets us view the true official rendering of the
# official BEHAVIOR data in a browser (matches the paper's project-page viz).
# Run in PointWorld root (main branch) with pw_extra_site on PYTHONPATH.
import sys
import time
from pathlib import Path

import numpy as np
import torch

from arguments import parse_args
from evaluation.tester import Tester
from utils import resolve_default_robot_urdf


def main():
    args = parse_args()
    tester = Tester(args)  # builds frozen model + applies checkpoint
    split = "test"

    from visualization.prediction_viz import (
        PredictionVisualizer, PredictionVisualizerConfig, build_sample_from_dictionary,
    )
    dl, _ = tester._build_eval_loader(split, enable_mask=False)
    viz_config = PredictionVisualizerConfig.from_args(args)
    urdf_path = Path(resolve_default_robot_urdf(args.domains))
    visualizer = PredictionVisualizer(viz_config, urdf_path=urdf_path)

    # scan up to SCAN batches, pick the sample whose GT scene flow moves the MOST (a real
    # manipulation window), so the viewer shows an object actually being moved.
    import os
    SCAN = int(os.environ.get("VIZ_SCAN", "40"))
    best = None  # (motion, batch, i)
    scanned = 0
    for batch in dl:
        if scanned >= SCAN:
            break
        scanned += 1
        b = {k: (v.to(tester.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        gtf = b["gt_scene_flows"].detach().cpu().numpy()
        for i in range(gtf.shape[0]):
            mv = np.linalg.norm(gtf[i][-1] - gtf[i][0], axis=-1)
            score = float(np.quantile(mv, 0.99))  # top-1% point displacement
            if best is None or score > best[0]:
                best = (score, batch, i)
    print(f"[viz-official] scanned {scanned} batches; best GT motion (p99 disp) = {best[0]:.3f}", flush=True)

    live_session = None
    for _ in range(1):
        batch = {k: (v.to(tester.device) if isinstance(v, torch.Tensor) else v) for k, v in best[1].items()}
        with torch.no_grad():
            outputs = tester.model(batch, training=False)
        batch_np = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        gt = batch_np["gt_scene_flows"]
        pred = outputs["scene_flows"].detach().cpu().numpy()
        B = gt.shape[0]
        i = best[2]
        sample_dict = {}
        for key, value in batch_np.items():
            if isinstance(value, np.ndarray):
                sample_dict[key] = value.item() if value.ndim == 0 else (value[i] if value.shape[0] == B else value)
            elif isinstance(value, (list, tuple)):
                sample_dict[key] = value[i] if len(value) == B else value
            else:
                sample_dict[key] = value
        sample_dict["gt_scene_flows"] = gt[i]
        sample_dict["__key__"] = str(batch_np["__key__"][i])
        sample_dict["__domain__"] = str(batch_np["__domain__"][i])

        viz_sample = build_sample_from_dictionary(sample_dict=sample_dict,
                                                  predictions={"scene_flows": pred[i]})
        result = visualizer.visualize(viz_sample, launch_viewer=True, live_session=live_session)
        live_session = result.get("live_session")
        host, port = visualizer.viewer_endpoint()
        host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
        print(f"[viz-official] READY at http://{host}:{port} — sample {sample_dict['__key__']} "
              f"domain {sample_dict['__domain__']} — holding open (Ctrl-C to stop)", flush=True)
        break  # only the first sample; hold the viewer

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[viz-official] stopped.", flush=True)


if __name__ == "__main__":
    main()
