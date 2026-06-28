# C-Task Fix Report: image_window data-path bug

**Date:** 2026-06-29
**Author:** Claude Code (automated fix + verification)
**Branch:** main

---

## Summary

Two critical bugs fixed in `starVLA/dataloader/gr00t_lerobot/datasets.py`:

- **C-1 (hard crash):** `get_data_by_modality` raised `ValueError("Invalid modality: image_window")` — now dispatches to `get_video` with `modality="image_window"`.
- **C-2 (silent corruption):** `_get_delta_indices` used bare video key strings as dict keys, so `image_window` overwrote `video`'s delta indices for shared video keys. Fixed by scoping keys as `f"{modality}.{key}"`.
- **Data-dict collision:** `get_step_data` stored both `video` and `image_window` results under `data["video.primary_image"]`, last writer winning. Fixed by storing image_window under `data["image_window.video.primary_image"]`, with corresponding `_pack_sample` update.

---

## Edits

### C-2 Fix — `_get_delta_indices` (line ~1198)

**Before:**
```python
def _get_delta_indices(self) -> dict[str, np.ndarray]:
    """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
    delta_indices: dict[str, np.ndarray] = {}
    for config in self.modality_configs.values():
        for key in config.modality_keys:
            delta_indices[key] = np.array(config.delta_indices)
    return delta_indices
```

**After:**
```python
def _get_delta_indices(self) -> dict[str, np.ndarray]:
    """Restructure the delta indices to use modality.key as keys instead of just the modalities.

    Keys are scoped as "<modality>.<key>" so that different modalities sharing the
    same raw video key (e.g. "video" and "image_window" both use "video.primary_image")
    get independent delta-index arrays and do not overwrite each other.
    """
    delta_indices: dict[str, np.ndarray] = {}
    for modality, config in self.modality_configs.items():
        for key in config.modality_keys:
            delta_indices[f"{modality}.{key}"] = np.array(config.delta_indices)
    return delta_indices
```

Result: `delta_indices["video.video.primary_image"] = [0]` and `delta_indices["image_window.video.primary_image"] = [-1,0,1,2,3]` — distinct, no collision.

---

### C-2 Read-site fixes

All 4 `self.delta_indices[key]` read sites updated to `self.delta_indices[f"{modality}.{key}"]`:

1. **`LeRobotSingleDataset.get_video`** (~line 1638): Added `modality: str = "video"` param; reads `self.delta_indices[f"{modality}.{key}"]`.
2. **`get_state_or_action`** (~line 1732): Already had `modality` param; reads `self.delta_indices[f"{modality}.{key}"]`.
3. **`get_language`** (~line 1786): Added `modality: str = "language"` param; reads `self.delta_indices[f"{modality}.{key}"]`.
4. **`CachedLeRobotSingleDataset.get_video`** (~line 2000): Added `modality: str = "video"` param; reads `self.delta_indices[f"{modality}.{key}"]`.

---

### C-1 Fix — `get_data_by_modality` (~line 1846)

**Before:**
```python
if modality == "video":
    return self.get_video(trajectory_id, key, base_index)
elif modality == "state" or modality == "action":
    return self.get_state_or_action(trajectory_id, modality, key, base_index)
elif modality == "language":
    return self.get_language(trajectory_id, key, base_index)
else:
    raise ValueError(f"Invalid modality: {modality}")
```

**After:**
```python
if modality == "video":
    return self.get_video(trajectory_id, key, base_index, modality="video")
elif modality == "image_window":
    # [Geo-MemoryVLA C-1 fix] Dispatch image_window to get_video with its own
    # scoped delta_indices (e.g. [-1,0,1,2,3]) rather than raising ValueError.
    return self.get_video(trajectory_id, key, base_index, modality="image_window")
elif modality == "state" or modality == "action":
    return self.get_state_or_action(trajectory_id, modality, key, base_index)
elif modality == "language":
    return self.get_language(trajectory_id, key, base_index, modality="language")
else:
    raise ValueError(f"Invalid modality: {modality}")
```

---

### Data-dict key collision fix — `get_step_data` (both classes)

**Before:**
```python
for modality in self.modality_keys:
    for key in self.modality_keys[modality]:
        data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
```

**After:**
```python
for modality in self.modality_keys:
    for key in self.modality_keys[modality]:
        if modality == "image_window":
            data[f"image_window.{key}"] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        else:
            data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
```

Applied to both `LeRobotSingleDataset.get_step_data` and `CachedLeRobotSingleDataset.get_step_data`.

---

### `_pack_sample` update

**Before:**
```python
window_keys = self.modality_keys["image_window"]
num_frames = data[window_keys[0]].shape[0]
image_window = []
for f in range(num_frames):
    frame_views = []
    for video_key in window_keys:
        img = Image.fromarray(data[video_key][f]).resize((224, 224))
```

**After:**
```python
window_keys = self.modality_keys["image_window"]
first_window_key = f"image_window.{window_keys[0]}"
num_frames = data[first_window_key].shape[0]
image_window = []
for f in range(num_frames):
    frame_views = []
    for video_key in window_keys:
        img = Image.fromarray(data[f"image_window.{video_key}"][f]).resize((224, 224))
```

---

### Test updates — `_fake_data` in `tests/geomemvla/test_image_window.py`

The `_fake_data` helper was updated to reflect the new key naming: video data at `"video.primary_image"` (1 frame), window data at `"image_window.video.primary_image"` (F frames) when `with_window=True`.

---

## New Integration Tests (3 added)

1. **`test_delta_indices_no_collision_with_image_window`**: Constructs real modality_configs from `Libero4in1DataConfig`, calls `_get_delta_indices`, asserts both `"video.video.primary_image"` → `[0]` and `"image_window.video.primary_image"` → `[-1,0,1,2,3]` exist as distinct entries.

2. **`test_get_data_by_modality_no_crash_for_image_window`**: Stubs `get_video` and calls `get_data_by_modality` with `modality="image_window"` — asserts no `ValueError` and the stub is invoked.

3. **`test_real_getitem_image_window_no_collision`**: Real `LeRobotSingleDataset.__getitem__` using LIBERO spatial data. Asserts: `sample["image"]` has 2 PIL images (1 per view, 224×224); `sample["image_window"]` has 5 frames each with 2 views; both are independent objects.

---

## Real LIBERO Data

Available at: `playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/`
Format: v2.0, 432 episodes, av1-encoded mp4 videos (backend: torchvision_av).
End-to-end `__getitem__` test ran successfully on real data.

---

## Verification Output

### Fast suite (`-m "not slow"`)
```
27 passed, 2 deselected, 2 warnings in 11.49s
```

### GPU smoke test (`CUDA_VISIBLE_DEVICES=1`)
```
test_forward_and_predict_geo_only  PASSED  [smoke] action_loss = 1.408505
test_geo_only_imagination_forward  PASSED  [smoke] action_loss = 1.432808, imagination_loss = 0.325544
2 passed in 222.70s
```

---

## Concerns

None. All 3 bug categories (C-1 dispatch crash, C-2 delta-index collision, data-dict key collision) are fixed with minimal surface area. The fix is backward-compatible: code paths that don't use `image_window` are unaffected since `get_video` and `get_language` default to `modality="video"` / `modality="language"` matching the old scoped keys.
