import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        # [Geo-MemoryVLA] Gate the image_window modality: declare it only when BOTH the
        # dataloader switch and imagination are on. ROBOT_TYPE_CONFIG_MAP shares the
        # DataConfig instance, so setting the attr here reaches modality_config().
        try:
            from examples.LIBERO.train_files.data_registry.data_config import ROBOT_TYPE_CONFIG_MAP
            _enable_win = bool(cfg.datasets.vla_data.get("enable_image_window", True)) and \
                          bool(cfg.framework.imagination.get("enabled", False))
            # [Geo-MemoryVLA] keep the window length in sync with the imagination config so the
            # dataloader window (context+chunk+1) matches the world model's context/chunk.
            _imag = cfg.framework.get("imagination", {}) if hasattr(cfg, "framework") else {}
            _ctx = int(_imag.get("context_size", 2))
            _chunk = int(_imag.get("horizon", 2))
            for _dc in ROBOT_TYPE_CONFIG_MAP.values():
                if hasattr(_dc, "enable_image_window"):
                    _dc.enable_image_window = _enable_win
                    if hasattr(_dc, "image_window_context"):
                        _dc.image_window_context = _ctx
                    if hasattr(_dc, "image_window_chunk"):
                        _dc.image_window_chunk = _chunk
        except (ImportError, AttributeError, KeyError) as _e:
            # [Geo-MemoryVLA] non-LIBERO runs / missing keys: leave DataConfig defaults.
            # Narrowed (not bare Exception) so real mistakes surface instead of silently
            # mis-gating the image_window modality.
            logger.warning(f"[Geo-MemoryVLA] image_window gate skipped: {_e}")

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )

        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
