# Task 4 Report: GeoMemoryVLA Hardening (horizon default + 3 guards)

**Date:** 2026-06-29
**Commit:** `24ac747`
**Branch:** `starVLA_dev`

---

## 3 Edits Applied

### Edit (a): Default horizon 4 → 2 (`GeoMemoryVLA.py` line 45-47)

**Before:**
```python
    imagination: dict = field(default_factory=lambda: {
        "enabled": True, "horizon": 4, "depth": 4, "steps": 4, "loss_scale": 0.5,
    })
```

**After:**
```python
    imagination: dict = field(default_factory=lambda: {
        # [Geo-MemoryVLA] horizon=2 (paper m=2, VGGT-World arXiv 2603.12655). Window =
        # context+horizon+1 = 5 frames. See image-window design spec.
        "enabled": True, "horizon": 2, "depth": 4, "steps": 4, "loss_scale": 0.5,
    })
```

### Edit (b): Construct-time window-length guard (`__init__`, after building `self.imaginer`)

**Before:** nothing after `self.imaginer = ImaginationAdapter(...)` block  
**After:** (inserted after the `ImaginationAdapter` constructor call)
```python
            # [Geo-MemoryVLA] Window must satisfy VGGTWorldModel Stage-2: context+chunk+1.
            ctx = int(fw.imagination.get("context_size", 2))
            chunk = int(fw.imagination["horizon"])
            self._imag_window_len = ctx + chunk + 1
            # (the dataloader derives the same length from the DataConfig; this is a guard.)
```

### Edit (c): `_build_image_window` — warn-once → hard raise

**Before:** (warn-once + degenerate fallback when `use_imag` is True)
```python
    def _build_image_window(self, examples, device):
        # ...
        for e in examples:
            if "image_window" not in e:
                if not getattr(self, "_warned_single_frame_window", False):
                    warnings.warn(
                        "GeoMemoryVLA: imagination enabled but no 'image_window' in batch; ..."
                    )
                    self._warned_single_frame_window = True
            frames = e.get("image_window", [e["image"]])
            ...
        return torch.stack(windows, dim=0).to(device)
```

**After:** (hard raise when `use_imag=True` and `image_window` absent; imagination-OFF keeps fallback)
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

Note: `import warnings` was left in place (harmless; used in other potential future paths).

---

## Test Results

### File tests (all 9):

```
PYTHONPATH=/workspace/tingting/starVLA NO_ALBUMENTATIONS_UPDATE=1 \
  /workspace/ghsun/miniconda3/envs/starVLA/bin/python \
  -m pytest tests/geomemvla/test_image_window.py -v
```

```
tests/geomemvla/test_image_window.py::test_image_window_indices_derivation PASSED
tests/geomemvla/test_image_window.py::test_modality_config_includes_window_when_enabled PASSED
tests/geomemvla/test_image_window.py::test_modality_config_omits_window_when_disabled PASSED
tests/geomemvla/test_image_window.py::test_pack_sample_emits_window_when_modality_present PASSED
tests/geomemvla/test_image_window.py::test_pack_sample_omits_window_when_modality_absent PASSED
tests/geomemvla/test_image_window.py::test_episode_boundary_clamp_keeps_past_in_episode PASSED
tests/geomemvla/test_image_window.py::test_episode_boundary_clamp_at_tail PASSED
tests/geomemvla/test_image_window.py::test_imagination_default_horizon_is_two PASSED
tests/geomemvla/test_image_window.py::test_build_image_window_raises_when_imag_on_and_window_missing PASSED
========================= 9 passed, 1 warning in 7.54s =========================
```

### Full fast geomemvla suite (22/22):

```
PYTHONPATH=/workspace/tingting/starVLA NO_ALBUMENTATIONS_UPDATE=1 \
  /workspace/ghsun/miniconda3/envs/starVLA/bin/python \
  -m pytest tests/geomemvla/ -v -m "not slow"
```

```
22 passed, 1 deselected, 1 warning in 10.08s
```

All framework-registration, config-load, condition-assembler, memory-bank, world-state-adapter, imagination-adapter, dataloader-episode-fields, and image-window tests pass.

---

## Commit

```
[starVLA_dev 24ac747] [Geo-MemoryVLA] horizon default 2; hard-raise on missing window; window-len guard
 2 files changed, 44 insertions(+), 16 deletions(-
```

---

## Concerns

- `import warnings` remains at the top of `GeoMemoryVLA.py` even though the warn-once call was removed. It is harmless — removing it would be a cosmetic cleanup that can be done with Task 5/6 if desired, but the brief says "leave it — harmless".
- The construct-time guard (`self._imag_window_len = ctx + chunk + 1`) currently only stores the derived value; it does not yet validate against a dataloader-reported window size. That cross-check is deferred to Task 5 (where the dataloader `DataConfig` is the source of truth) per the spec.
- The imagination-OFF path in `_build_image_window` retains the single-frame fallback (no raise). This is intentional per the brief: when `use_imag=False`, degenerating is safe (the imaginer is never called).

---

## Comment/import fix (review finding follow-up, 2026-06-29)

### Finding 1 — stale `forward()` comment in `GeoMemoryVLA.py`

**Before (lines 233-236):**
```python
            # The vendored world model consumes a multi-frame image window. When the
            # dataloader provides one ("image_window" key: context+chunk frames), pass it;
            # otherwise fall back to the current views (degenerate single-step, Task 5 note).
            # `where` drives the stage-1/stage-2 flow-forcing curriculum.
```

**After:**
```python
            # [Geo-MemoryVLA] The vendored world model needs a multi-frame window. The
            # dataloader must supply "image_window" (context+chunk+1 frames) when imagination
            # is on; _build_image_window raises ValueError if it is missing (no silent
            # single-frame degeneration). `where` drives stage-1/stage-2 flow-forcing.
```

### Finding 2 — mid-file `import pytest` in `tests/geomemvla/test_image_window.py`

**Before:** `import pytest` appeared at line 112 (after `test_episode_boundary_clamp_at_tail`), mid-file.  
**After:** `import pytest` moved to line 7 (top of file, after `import importlib`). Mid-file occurrence removed.

### Verification outputs

```
grep -n "import pytest" tests/geomemvla/test_image_window.py
→ 7:import pytest     (exactly once, at top)

python -c "import ast; ast.parse(open('starVLA/model/framework/VLM4A/GeoMemoryVLA.py').read()); print('parse OK')"
→ parse OK

pytest tests/geomemvla/test_image_window.py -v
→ 9 passed, 1 warning in 7.18s

pytest tests/geomemvla/ -v -m "not slow"
→ 22 passed, 1 deselected, 1 warning in 10.79s
```
