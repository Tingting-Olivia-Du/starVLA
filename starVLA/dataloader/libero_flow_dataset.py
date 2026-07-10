# [LangPointWorld] S1-b: self-contained LIBERO + teacher-flow-cache dataset. Reads directly from the
# raw LIBERO HDF5 (RGB agentview+wrist, delta actions, proprio) and the matching teacher-flow cache
# (pw_build_teacher_cache.py output), keyed by (task, demo) — SAME source, so cache alignment is
# exact-by-construction (avoids the fragile LeRobot<->HDF5 episode match). Emits the example dict
# QwenPIFlow.forward consumes: {image:[agentview,wrist], lang, action[T,7], state[1,7],
# teacher_flow[N,Hf,3], teacher_weight[N], frame_id}.
import glob
import os

import numpy as np
import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"


def _task_instruction(task):
    return f"pick up the {task.replace('_', ' ')} and place it in the basket"


class LiberoFlowDataset(Dataset):
    def __init__(self, cache_dir, action_horizon=8, img_size=None, split="train",
                 val_frac=0.1, seed=42):
        self.cache_files = sorted(glob.glob(os.path.join(cache_dir, "*__demo_*.npz")))
        assert self.cache_files, f"no cache in {cache_dir}"
        # deterministic train/val split over cache entries
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.cache_files))
        n_val = max(1, int(len(self.cache_files) * val_frac))
        val_idx = set(perm[:n_val].tolist())
        keep = [i for i in range(len(self.cache_files)) if (i in val_idx) == (split == "val")]
        self.cache_files = [self.cache_files[i] for i in keep]
        self.action_horizon = action_horizon
        self.img_size = img_size
        self._h5cache = {}
        print(f"[LiberoFlowDataset] split={split} n={len(self.cache_files)}", flush=True)

    def __len__(self):
        return len(self.cache_files)

    def _h5(self, task):
        if task not in self._h5cache:
            self._h5cache[task] = h5py.File(f"{DATA}/{_task_instruction(task).replace(' ', '_')}_demo.hdf5", "r")
        return self._h5cache[task]

    def _img(self, arr):
        im = Image.fromarray(arr[::-1])  # LIBERO agentview stored upside-down for display convention
        if self.img_size:
            im = im.resize((self.img_size, self.img_size))
        return im

    def __getitem__(self, i):
        z = np.load(self.cache_files[i], allow_pickle=True)
        task = str(z["task"]); demo = str(z["demo"]); ti = int(z["frame_id"])
        f = self._h5(task); d = f["data"][f"demo_{demo}"]
        # action chunk starting at the cache timestep ti (H steps of the demo delta actions)
        acts = np.asarray(d["actions"][:], np.float32)                 # [L,7] in [-1,1]
        L = acts.shape[0]
        chunk = acts[ti: ti + self.action_horizon]
        if chunk.shape[0] < self.action_horizon:                       # pad tail with last action
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.action_horizon - chunk.shape[0], 0)], 0)
        state = np.asarray(d["obs"]["ee_states"][ti], np.float32)[:7]   # ee 6d + pad; use 7 dims
        if state.shape[0] < 7:
            state = np.concatenate([state, np.zeros(7 - state.shape[0], np.float32)])
        av = np.asarray(z["agentview_rgb"]); wr = np.asarray(z["wrist_rgb"])
        return {
            "image": [self._img(av), self._img(wr)],
            "lang": _task_instruction(task),
            "action": chunk.astype(np.float32),                        # [H,7]
            "state": state[None].astype(np.float32),                   # [1,7]
            "teacher_flow": np.asarray(z["F_teacher"], np.float32),     # [N,Hf,3]
            "teacher_weight": np.asarray(z["teacher_weight"], np.float32),  # [N]
            "frame_id": ti,
        }


def collate_examples(batch):
    """Framework forward takes a list of example dicts; keep it as-is."""
    return batch
