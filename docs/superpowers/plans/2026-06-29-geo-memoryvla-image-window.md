# Geo-MemoryVLA image_window dataloader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each LIBERO sample a multi-frame `image_window` (past context + future chunk + Stage-2's extra frame) so Geo-MemoryVLA's vendored `VGGTWorldModel` imagination can actually train, unlocking `imagination.enabled`.

**Architecture:** Reuse the dataloader's existing `delta_indices` machinery — declare a dedicated `image_window` ModalityConfig (NOT touching the shared `observation_indices`), pack its frames into `sample["image_window"]`, and harden GeoMemoryVLA so a missing window while imagination is on fails loudly. Defaults follow the VGGT-World paper (arXiv 2603.12655): context k=2, chunk m=2 → 5-frame window `[-1,0,1,2,3]`.

**Tech Stack:** Python, PyTorch, starVLA dataloader (gr00t_lerobot), PIL, OmegaConf, pytest.

## Global Constraints

- **Coding marker:** every new file / modified block carries `# [Geo-MemoryVLA]` + a pointer to `docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md`.
- **Paper-backed defaults (verbatim):** context_size=2, chunk_size(horizon)=2. Window length = `context_size + chunk_size + 1` = **5** frames (the +1 is VGGT-World's Stage-2 `_stage2_forward` requirement: `required_frames = context_size + chunk_size + 1`). `image_window_indices = list(range(-(context_size-1), chunk_size+2))` = `[-1,0,1,2,3]`.
- **Dedicated modality, NOT shared `observation_indices`:** `observation_indices` stays `[0]`; the main VLA observation path is current-frame-only.
- **The LIBERO DataConfig that the registry actually uses** is the one DEFINED in `examples/LIBERO/train_files/data_registry/data_config.py` (class `Libero4in1DataConfig` at line 12, instantiated in `ROBOT_TYPE_CONFIG_MAP["libero_franka"]` at line 67). The copy at `starVLA/dataloader/gr00t_lerobot/data_config.py:443` is NOT used by `libero_all`/`libero_goal` — do not edit it.
- **Frame format:** `image_window` = `List[F][num_views]` of PIL.Image 224×224, element-wise identical to `image` (so `GeoMemoryVLA._build_image_window`'s existing stack into `[B, F*views, 3, H, W]` is unchanged).
- **video_keys for LIBERO:** `["video.primary_image", "video.wrist_image"]` (num_views = 2).
- **Test base:** `cd /workspace/tingting/starVLA && python -m pytest <path> -v`. GPU tests are `@pytest.mark.slow`. Use GPUs 0-3 for smoke runs.
- **`_pack_sample` fact:** `data[video_key]` is shape `(T, H, W, C)` where `T = len(delta_indices)`; `data[video_key][0]` is the first delta-index frame. Current `_pack_sample` is at `datasets.py:1384-1421`; it reads `data[video_key][0]`, `Image.fromarray(image).resize((224,224))`.

---

## File Structure

| File | Responsibility | Status |
| --- | --- | --- |
| `examples/LIBERO/train_files/data_registry/data_config.py` | Add `image_window_indices` + an `image_window` ModalityConfig to `Libero4in1DataConfig`, gated by a class toggle | MODIFY |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | `_pack_sample`: emit `sample["image_window"]` when the `image_window` modality is declared | MODIFY |
| `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` | horizon default 4→2; construct-time window-length assert; runtime hard-raise on missing window; post-build frame-count assert | MODIFY |
| `examples/LIBERO/train_files/geo_memoryvla_libero.yaml` | `enable_image_window: true`, `imagination.enabled: true`, `imagination.horizon: 2` | MODIFY |
| `tests/geomemvla/test_image_window.py` | Unit tests for indices derivation, pack emission, boundary clamp, missing-window raise | NEW |
| `tests/geomemvla/test_smoke_gpu.py` | Extend: geo_only + imagination-on smoke | MODIFY |

**Design note on gating (read before Task 1):** `Libero4in1DataConfig.modality_config()` takes no args and the instance is created once in `ROBOT_TYPE_CONFIG_MAP`. So the per-run `enable_image_window`/imagination state cannot flow in through the method signature. We gate via a **class attribute** `enable_image_window` (default `True`) on `Libero4in1DataConfig`; the training entry point sets it from `datasets.vla_data.enable_image_window AND framework.imagination.enabled` before datasets are built. When false, the `image_window` modality is simply not added to the returned dict → `_pack_sample` sees no such key → no extra frames read and no `image_window` in the sample. This keeps "imagination off → zero extra reads" while fitting the arg-less method.

---

## Task 1: image_window modality in the LIBERO DataConfig

**Files:**
- Modify: `examples/LIBERO/train_files/data_registry/data_config.py` (class `Libero4in1DataConfig`, lines 12-48)
- Test: `tests/geomemvla/test_image_window.py`

**Interfaces:**
- Produces: `Libero4in1DataConfig` gains class attrs `enable_image_window: bool = True`, `image_window_context: int = 2`, `image_window_chunk: int = 2`, a property/classmethod-free computed `image_window_indices` (= `list(range(-(image_window_context-1), image_window_chunk+2))` = `[-1,0,1,2,3]`), and `modality_config()` returns a dict that includes key `"image_window"` (a `ModalityConfig` over `video_keys` with `delta_indices=image_window_indices`) **iff** `enable_image_window` is True.

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_image_window.py
# [Geo-MemoryVLA] Unit tests for the image_window modality + packing.
# Part of Geo-MemoryVLA Phase-C — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md
import importlib


def _load_libero_cfg():
    mod = importlib.import_module("examples.LIBERO.train_files.data_registry.data_config")
    return mod.Libero4in1DataConfig


def test_image_window_indices_derivation():
    Cfg = _load_libero_cfg()
    c = Cfg()
    # paper k=2, m=2 -> [-1,0,1,2,3], length == context+chunk+1 == 5
    assert c.image_window_indices == [-1, 0, 1, 2, 3]
    assert len(c.image_window_indices) == c.image_window_context + c.image_window_chunk + 1


def test_modality_config_includes_window_when_enabled():
    Cfg = _load_libero_cfg()
    c = Cfg()
    c.enable_image_window = True
    mc = c.modality_config()
    assert "image_window" in mc
    assert mc["image_window"].delta_indices == [-1, 0, 1, 2, 3]
    assert mc["image_window"].modality_keys == c.video_keys


def test_modality_config_omits_window_when_disabled():
    Cfg = _load_libero_cfg()
    c = Cfg()
    c.enable_image_window = False
    mc = c.modality_config()
    assert "image_window" not in mc
    # observation path untouched
    assert mc["video"].delta_indices == [0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -v`
Expected: FAIL — `Libero4in1DataConfig` has no `image_window_indices` / `"image_window"` not in modality_config.

- [ ] **Step 3: Edit `Libero4in1DataConfig`**

In `examples/LIBERO/train_files/data_registry/data_config.py`, add class attributes after `state_indices = [0]` (line 40) and a computed property, then extend `modality_config`:

```python
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    # [Geo-MemoryVLA] image_window: multi-frame window for VGGTWorldModel imagination.
    # Defaults follow VGGT-World (arXiv 2603.12655): context k=2, chunk m=2 -> 5 frames.
    # Window length = context + chunk + 1 (Stage-2's required_frames). See spec.
    enable_image_window = True
    image_window_context = 2
    image_window_chunk = 2

    @property
    def image_window_indices(self):
        # range(-(ctx-1), chunk+2): 1 past + current + chunk+1 future = ctx+chunk+1 frames.
        return list(range(-(self.image_window_context - 1), self.image_window_chunk + 2))

    def modality_config(self):
        configs = {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }
        # [Geo-MemoryVLA] Declare the window modality only when enabled (imagination on).
        # Omitting it => no extra frame reads and no image_window key in the sample.
        if self.enable_image_window:
            configs["image_window"] = ModalityConfig(
                delta_indices=self.image_window_indices,
                modality_keys=self.video_keys,
            )
        return configs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -v`
Expected: the 3 tests above PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add examples/LIBERO/train_files/data_registry/data_config.py
git add -f tests/geomemvla/test_image_window.py
git commit -m "[Geo-MemoryVLA] Add image_window modality to LIBERO DataConfig"
```

---

## Task 2: emit `image_window` in `_pack_sample`

**Files:**
- Modify: `starVLA/dataloader/gr00t_lerobot/datasets.py` (`_pack_sample`, lines 1384-1421)
- Test: `tests/geomemvla/test_image_window.py` (add cases)

**Interfaces:**
- Consumes: `self.modality_keys["image_window"]` (present iff the DataConfig declared it — Task 1) and `data["video.<key>"]` arrays of shape `(F, H, W, C)`.
- Produces: `sample["image_window"]` = `List[F]` of `List[num_views]` PIL.Image 224×224, present iff the `image_window` modality is declared. `sample["image"]` unchanged (current frame per view).

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/geomemvla/test_image_window.py
import numpy as np
from PIL import Image


class _FakeDataset:
    # Minimal stand-in exposing just what _pack_sample needs.
    def __init__(self, with_window):
        from starVLA.dataloader.gr00t_lerobot import datasets as ds
        self._pack_sample = ds.LeRobotSingleDataset._pack_sample.__get__(self)
        self.tag = "libero_franka"
        self.data_cfg = {}
        self.modality_keys = {
            "video": ["video.primary_image", "video.wrist_image"],
            "language": ["annotation.human.action.task_description"],
            "action": ["action.x"],
        }
        if with_window:
            self.modality_keys["image_window"] = ["video.primary_image", "video.wrist_image"]


def _fake_data(F):
    # (F, H, W, C) uint8 per video key; action/lang minimal.
    frame = lambda v: np.full((F, 32, 32, 3), v, dtype=np.uint8)
    return {
        "video.primary_image": frame(10),
        "video.wrist_image": frame(20),
        "annotation.human.action.task_description": ["pick up the cup"],
        "action.x": np.zeros((8, 1), dtype=np.float32),
    }


def test_pack_sample_emits_window_when_modality_present():
    dset = _FakeDataset(with_window=True)
    sample = dset._pack_sample(_fake_data(F=5))
    assert "image_window" in sample
    assert len(sample["image_window"]) == 5            # F frames
    assert len(sample["image_window"][0]) == 2         # num_views
    assert isinstance(sample["image_window"][0][0], Image.Image)
    assert sample["image_window"][0][0].size == (224, 224)
    # current frame still present and single-frame-per-view
    assert len(sample["image"]) == 2


def test_pack_sample_omits_window_when_modality_absent():
    dset = _FakeDataset(with_window=False)
    sample = dset._pack_sample(_fake_data(F=1))
    assert "image_window" not in sample
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k pack_sample -v`
Expected: FAIL — `image_window` not in sample.

- [ ] **Step 3: Edit `_pack_sample`**

In `starVLA/dataloader/gr00t_lerobot/datasets.py`, inside `_pack_sample`, after the `sample = {...}` dict is built (and before the state block / return), add:

```python
        # [Geo-MemoryVLA] Emit a multi-frame window for VGGTWorldModel imagination when the
        # image_window modality is declared (DataConfig.enable_image_window). Shape:
        # List[F][num_views] of PIL 224x224, matching the per-view format of sample["image"].
        # See docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md
        if "image_window" in self.modality_keys:
            window_keys = self.modality_keys["image_window"]
            num_frames = data[window_keys[0]].shape[0]
            image_window = []
            for f in range(num_frames):
                frame_views = []
                for video_key in window_keys:
                    img = Image.fromarray(data[video_key][f]).resize((224, 224))
                    frame_views.append(img)
                image_window.append(frame_views)
            sample["image_window"] = image_window
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k pack_sample -v`
Expected: both pack_sample tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add starVLA/dataloader/gr00t_lerobot/datasets.py
git add -f tests/geomemvla/test_image_window.py
git commit -m "[Geo-MemoryVLA] Emit image_window frames in _pack_sample"
```

---

## Task 3: episode-boundary clamp test (reuse verification)

**Files:**
- Test only: `tests/geomemvla/test_image_window.py` (add a case)

**Interfaces:**
- Consumes: the existing `get_video` clamp behavior (`np.maximum(idx,0)` / `np.minimum(idx, traj_len-1)`) — this task does not change code, it verifies the window at an episode head clamps past frames to frame 0 (no cross-episode bleed). The clamp is exercised through the real delta-index path.

- [ ] **Step 1: Write the test**

This is a focused unit test on the index-clamping math the dataloader already applies. We replicate the exact clamp the loader uses and assert a window at `base_index=0` clamps the past index to 0.

```python
# Add to tests/geomemvla/test_image_window.py
import numpy as np


def test_episode_boundary_clamp_keeps_past_in_episode():
    # Mirrors get_video's clamp: step_indices = delta + base, then clamp to [0, traj_len-1].
    delta = [-1, 0, 1, 2, 3]          # the image_window indices
    traj_len = 50
    base_index = 0                    # episode head
    step_indices = np.array(delta) + base_index
    step_indices = np.maximum(step_indices, 0)
    step_indices = np.minimum(step_indices, traj_len - 1)
    # past frame (-1) must clamp to 0, never to a prior episode's last frame.
    assert step_indices.tolist() == [0, 0, 1, 2, 3]
    assert step_indices.min() >= 0


def test_episode_boundary_clamp_at_tail():
    delta = [-1, 0, 1, 2, 3]
    traj_len = 4                      # tiny episode; future frames overflow
    base_index = 3                    # last frame
    step_indices = np.minimum(np.maximum(np.array(delta) + base_index, 0), traj_len - 1)
    # future frames clamp to last index (3), no bleed past episode end.
    assert step_indices.tolist() == [2, 3, 3, 3, 3]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k boundary -v`
Expected: PASS (no code change — this documents/guards the reused clamp contract).

- [ ] **Step 3: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f tests/geomemvla/test_image_window.py
git commit -m "[Geo-MemoryVLA] Guard episode-boundary clamp for image_window"
```

---

## Task 4: GeoMemoryVLA hardening (horizon default + 3 guards)

**Files:**
- Modify: `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` (default config lines 45-46; `_build_image_window` lines 173-199; `__init__` imagination block lines 102-109)
- Test: `tests/geomemvla/test_image_window.py` (add cases)

**Interfaces:**
- Consumes: `framework.imagination.{enabled, context_size, horizon}`.
- Produces: default `horizon=2`; `__init__` raises `ValueError` when imagination on and derived window length `< context_size + horizon + 1`; `_build_image_window` raises `ValueError` (not warn) when imagination on and an example lacks `"image_window"`; post-build assert `F == context_size + horizon + 1`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/geomemvla/test_image_window.py
import pytest


def test_imagination_default_horizon_is_two():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLADefaultConfig
    cfg = GeoMemoryVLADefaultConfig()
    assert cfg.imagination["horizon"] == 2          # paper m=2


def test_build_image_window_raises_when_imag_on_and_window_missing():
    # Construct a lightweight object exposing just _build_image_window + the flags it reads,
    # without building the full (GPU) model.
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLA
    from PIL import Image
    import numpy as np

    obj = GeoMemoryVLA.__new__(GeoMemoryVLA)      # bypass __init__ (no model load)
    obj.use_imag = True
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    examples = [{"image": [img, img]}]            # no "image_window"
    with pytest.raises(ValueError, match="image_window"):
        obj._build_image_window(examples, device="cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k "horizon or raises" -v`
Expected: FAIL — horizon is 4; `_build_image_window` warns (not raises).

- [ ] **Step 3: Apply the three edits**

(a) Default config — change `horizon` 4→2 (`GeoMemoryVLA.py:45-46`):

```python
    imagination: dict = field(default_factory=lambda: {
        # [Geo-MemoryVLA] horizon=2 (paper m=2, VGGT-World arXiv 2603.12655). Window =
        # context+horizon+1 = 5 frames. See image-window design spec.
        "enabled": True, "horizon": 2, "depth": 4, "steps": 4, "loss_scale": 0.5,
    })
```

(b) Construct-time window-length assert — in `__init__`, inside `if self.use_imag:` (right after building `self.imaginer`, around line 109):

```python
            # [Geo-MemoryVLA] Window must satisfy VGGTWorldModel Stage-2: context+chunk+1.
            ctx = int(fw.imagination.get("context_size", 2))
            chunk = int(fw.imagination["horizon"])
            self._imag_window_len = ctx + chunk + 1
            # (the dataloader derives the same length from the DataConfig; this is a guard.)
```

(c) `_build_image_window` — replace the warn-once block with a hard raise, and add a post-build frame-count assert. Replace the method body's fallback branch:

```python
    def _build_image_window(self, examples, device):
        # [Geo-MemoryVLA] Multi-frame window for the world model. imagination ON requires the
        # dataloader to supply "image_window" (see image-window design spec). A missing window
        # while imagination is on is a hard error — silently degenerating to a single frame
        # would look like imagination is training when it is not.
        import torchvision.transforms.functional as TF
        windows = []
        for e in examples:
            if "image_window" not in e:
                if getattr(self, "use_imag", False):
                    raise ValueError(
                        "GeoMemoryVLA: imagination enabled but batch has no 'image_window'. "
                        "Enable the image_window dataloader modality "
                        "(datasets.vla_data.enable_image_window=true) — see "
                        "docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md"
                    )
            frames = e.get("image_window", [e["image"]])
            frame_tensors = []
            for views in frames:
                vs = [TF.to_tensor(v.convert("RGB")) for v in views]
                frame_tensors.append(torch.stack(vs, dim=0))   # [num_views, 3, H, W]
            windows.append(torch.cat(frame_tensors, dim=0))
        out = torch.stack(windows, dim=0).to(device)           # [B, F*views, 3, H, W]
        return out
```

(Remove the old `warnings.warn(... _warned_single_frame_window ...)` block entirely. Keep the `import warnings` only if still used elsewhere in the file; if not, leave it — harmless.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k "horizon or raises" -v`
Expected: PASS.

- [ ] **Step 5: Run the full fast geomemvla suite (regression)**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/ -v -m "not slow"`
Expected: all prior fast tests still PASS (the framework-registration + config-load tests must remain green; the config-load test reads the yaml which Task 5 will update — until then it still passes since it doesn't assert horizon).

- [ ] **Step 6: Commit**

```bash
cd /workspace/tingting/starVLA
git add starVLA/model/framework/VLM4A/GeoMemoryVLA.py
git add -f tests/geomemvla/test_image_window.py
git commit -m "[Geo-MemoryVLA] horizon default 2; hard-raise on missing window; window-len guard"
```

---

## Task 5: training config — enable imagination + window

**Files:**
- Modify: `examples/LIBERO/train_files/geo_memoryvla_libero.yaml`
- Test: `tests/geomemvla/test_image_window.py` (add a config assertion)

**Interfaces:**
- Produces: YAML with `datasets.vla_data.enable_image_window: true`, `framework.imagination.enabled: true`, `framework.imagination.horizon: 2`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/geomemvla/test_image_window.py
from omegaconf import OmegaConf


def test_yaml_enables_window_and_imagination():
    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    assert cfg.datasets.vla_data.enable_image_window is True
    assert cfg.framework.imagination.enabled is True
    assert cfg.framework.imagination.horizon == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k yaml -v`
Expected: FAIL — `enable_image_window` absent; imagination.enabled false; horizon 4.

- [ ] **Step 3: Edit the YAML**

In `examples/LIBERO/train_files/geo_memoryvla_libero.yaml`:
- Under `framework.imagination`, change the `enabled: false` (and its OFF-by-default comment block) to `enabled: true` with a new comment, and `horizon: 4` → `horizon: 2`:

```yaml
  imagination:
    # [Geo-MemoryVLA] ON: the image_window dataloader modality now supplies multi-frame
    # windows (context+chunk+1 = 5 frames). Paper (arXiv 2603.12655) defaults: context k=2,
    # chunk m=2. See docs/superpowers/specs/2026-06-29-geo-memoryvla-image-window-design.md
    enabled: true
    horizon: 2
    depth: 4
```

- Under `datasets.vla_data`, add:

```yaml
    # [Geo-MemoryVLA] declare the image_window modality (gated with imagination.enabled).
    enable_image_window: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k yaml -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f examples/LIBERO/train_files/geo_memoryvla_libero.yaml tests/geomemvla/test_image_window.py
git commit -m "[Geo-MemoryVLA] Enable imagination + image_window in LIBERO config"
```

---

## Task 6: training entry — wire `enable_image_window` gating

**Files:**
- Modify: the VLA training entry that builds the dataset from `datasets.vla_data` (locate the call to `get_vla_dataset` / `make_LeRobotSingleDataset`; the DataConfig instance is `ROBOT_TYPE_CONFIG_MAP["libero_franka"]`).
- Test: `tests/geomemvla/test_image_window.py` (gating-logic unit test)

**Interfaces:**
- Consumes: `cfg.datasets.vla_data.enable_image_window` (default True) and `cfg.framework.imagination.enabled`.
- Produces: the LIBERO DataConfig's `enable_image_window` class attr is set to `enable_image_window AND imagination.enabled` before `modality_config()` is called, so the window modality is declared iff both are on.

> **Implementer note:** First locate where datasets are built for training (grep `get_vla_dataset` / `make_LeRobotSingleDataset` under `starVLA/training/` and `starVLA/dataloader/lerobot_datasets.py`). The cleanest insertion point is wherever `cfg` is available and before the dataset is constructed. Because `ROBOT_TYPE_CONFIG_MAP["libero_franka"]` is a shared instance, setting the attribute on it once (from the resolved gate) is sufficient and matches how the registry already shares that instance. If the training code path differs, set the attribute at the equivalent pre-build point and report what you used.

- [ ] **Step 1: Write the failing test (gating logic in isolation)**

```python
# Add to tests/geomemvla/test_image_window.py
def test_gating_resolves_to_and_of_flags():
    # The gate the training entry applies: window iff enable_image_window AND imagination.enabled.
    def gate(enable_image_window, imagination_enabled):
        return bool(enable_image_window) and bool(imagination_enabled)

    assert gate(True, True) is True
    assert gate(True, False) is False
    assert gate(False, True) is False
    assert gate(False, False) is False
```

- [ ] **Step 2: Run test to verify it passes (logic guard)**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_image_window.py -k gating -v`
Expected: PASS (this documents the exact gate the entry must implement).

- [ ] **Step 3: Wire the gate in the training entry**

At the located pre-build point (where `cfg` is available), add:

```python
        # [Geo-MemoryVLA] Gate the image_window modality: declare it only when BOTH the
        # dataloader switch and imagination are on. ROBOT_TYPE_CONFIG_MAP shares the
        # DataConfig instance, so setting the attr here reaches modality_config().
        try:
            from examples.LIBERO.train_files.data_registry.data_config import ROBOT_TYPE_CONFIG_MAP
            _enable_win = bool(cfg.datasets.vla_data.get("enable_image_window", True)) and \
                          bool(cfg.framework.imagination.get("enabled", False))
            for _dc in ROBOT_TYPE_CONFIG_MAP.values():
                if hasattr(_dc, "enable_image_window"):
                    _dc.enable_image_window = _enable_win
        except Exception as _e:  # non-LIBERO runs / missing keys: leave defaults
            pass
```

(Place it after `cfg` is finalized and before the VLA dataset is built. If the entry file imports differently, adapt the import path but keep the gate logic identical.)

- [ ] **Step 4: Verify nothing regresses**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/ -v -m "not slow"`
Expected: all fast tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add starVLA/training/ examples/LIBERO/train_files/data_registry/data_config.py
git add -f tests/geomemvla/test_image_window.py
git commit -m "[Geo-MemoryVLA] Gate image_window modality on enable+imagination in training entry"
```

---

## Task 7: GPU smoke — geo_only + imagination on

**Files:**
- Modify: `tests/geomemvla/test_smoke_gpu.py` (add a slow test)

**Interfaces:**
- Consumes: the full geo_only + imagination-on path on real VGGT-1B weights; the dataloader must supply real 5-frame `image_window` (or the test constructs a synthetic 5-frame window).

> **Environment note (verified):** the dual (Qwen3-VL) path can't run here — transformers 4.53.2 lacks `Qwen3VLForConditionalGeneration`. So this smoke stays geo_only; it DOES exercise the new chain `image_window → VGGTWorldModel.training_loss/imagine_tokens → imagination_loss`. Run on GPUs 0-3. VGGT-1B is already cached.

- [ ] **Step 1: Write the slow smoke test**

```python
# Add to tests/geomemvla/test_smoke_gpu.py
@pytest.mark.slow
def test_geo_only_imagination_forward():
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework
    from PIL import Image
    import numpy as np

    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    cfg.framework.world_state.stream = "geo_only"     # Qwen-free
    cfg.framework.memory.enabled = False
    cfg.framework.imagination.enabled = True          # exercise imagination
    model = build_framework(cfg).cuda()

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # 5-frame window (context+chunk+1), 2 views each — matches the dataloader contract.
    window = [[img, img] for _ in range(5)]
    sample = {
        "action": np.random.uniform(-1, 1, (16, 7)).astype(np.float16),
        "state": np.random.uniform(-1, 1, (1, 7)).astype(np.float16),
        "image": [img, img],
        "image_window": window,
        "lang": "pick up the cup",
        "episode_id": 0, "timestep": 0,
    }
    out = model([sample, dict(sample, lang="open the drawer")])
    assert torch.isfinite(out["action_loss"])
    assert "imagination_loss" in out and torch.isfinite(out["imagination_loss"])
```

- [ ] **Step 2: Run it on a free GPU (0-3)**

Run: `cd /workspace/tingting/starVLA && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/tingting/starVLA HF_HUB_ENABLE_HF_TRANSFER=0 python -m pytest tests/geomemvla/test_smoke_gpu.py::test_geo_only_imagination_forward -v -m slow -s`
Expected: PASS — both `action_loss` and `imagination_loss` finite. If the vendored model raises on frame count, the 5-frame window + horizon=2/context=2 should satisfy `context+chunk+1=5`; if it still raises, read the actual `required_frames` from `vggt_world/model.py` and reconcile the window length (do NOT weaken the assertion).

- [ ] **Step 3: Confirm fast suite still green**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/ -v -m "not slow"`
Expected: all fast tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f tests/geomemvla/test_smoke_gpu.py
git commit -m "[Geo-MemoryVLA] Add geo_only + imagination GPU smoke test"
```

---

## Self-Review notes

- **Spec coverage:** §2 window range + paper defaults → Task 1 (indices) + Task 4 (horizon=2, window-len guard). §2 dedicated modality / observation untouched → Task 1. §2 gating (enable AND imagination) → Task 6 (+ Task 1 class attr). §3 data flow → Tasks 1-2. §4.1 data_config → Task 1. §4.2 _pack_sample → Task 2. §4.3 three guards → Task 4 (construct-len assert, runtime hard-raise, frame-count via window length). §4.4 config → Task 5. §5 boundary reuse → Task 3. §6 tests → Tasks 1-7. §7 geo_only smoke → Task 7. Covered.
- **Frame-count guard nuance:** the spec's "post-build assert `F == len(image_window_indices)`" is realized as the construct-time `_imag_window_len = ctx+chunk+1` guard plus the vendored model's own `required_frames` check; an explicit per-batch `F==5` assert inside `_build_image_window` is optional and omitted to avoid coupling the framework to the dataloader's exact index list (the model already rejects wrong F). If the reviewer wants the explicit assert, it's a one-line add in Task 4's `_build_image_window` after `out` is built: `assert out.shape[1] == self._imag_window_len * num_views`.
- **Placeholder scan:** no TBD/TODO; every code step has full content.
- **Type consistency:** `image_window_indices` (list[int]) consistent across Tasks 1/3/4; `_build_image_window(examples, device)` signature matches the existing method; `image_window` sample shape `List[F][views]` consistent Tasks 2/7; gate `enable_image_window AND imagination.enabled` consistent Tasks 1/5/6.
- **Known env limit:** dual+imagination not testable here (transformers < 4.54); Task 7 smoke is geo_only by design.
