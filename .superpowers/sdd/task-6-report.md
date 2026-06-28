# Task 6 Report: GeoMemoryVLA Framework Orchestrator

## Steps

1. Read brief at `.superpowers/sdd/task-6-brief.md`.
2. Explored existing framework structure: `starVLA/model/framework/VLM4A/` (QwenGR00T.py for skeleton), `share_tools.py`, `base_framework.py`, `starVLA/model/tools.py` for FRAMEWORK_REGISTRY.
3. Wrote failing tests at `tests/geomemvla/test_geomemvla_framework.py` (TDD step 1).
4. Ran tests — both failed as expected.
5. Implemented `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` using exact code from the brief.
6. Discovered `_auto_import_framework_modules()` in `base_framework.py` propagates exceptions from any failing sub-module import (pre-existing `QwenAdapter.py` fails with `ImportError: cannot import name 'Qwen3VLForConditionalGeneration'` in this environment's transformers version). Added try/except around each import call with a warning log.
7. Ran tests — both passed.
8. Ran all 12 `tests/geomemvla/` tests — all passed.
9. Committed with `git add -f`.

## Phase-C Concerns

### 1. Multi-frame image window

`_build_image_window` handles the multi-frame window correctly:

```python
def _build_image_window(self, examples, device):
    # Multi-frame window for the world model. Prefer an explicit "image_window"
    # (List[frames][views] of PIL) if the dataloader supplies it; else degenerate
    # to the current single frame's views (Task 5 note / Phase-C follow-up).
    import torchvision.transforms.functional as TF
    windows = []
    for e in examples:
        frames = e.get("image_window", [e["image"]])
        frame_tensors = []
        for views in frames:
            vs = [TF.to_tensor(v.convert("RGB")) for v in views]
            frame_tensors.append(torch.stack(vs, dim=0))   # [num_views, 3, H, W]
        # Flatten (frames, views) into the world model's frame axis.
        windows.append(torch.cat(frame_tensors, dim=0))
    return torch.stack(windows, dim=0).to(device)          # [B, F*views, 3, H, W]
```

The `# [Geo-MemoryVLA]` degeneration note is preserved exactly as the brief specifies.

### 2. `imag` stream_dim set to `geo_dim` (1024)

In the stream_dims wiring:
```python
if self.use_imag:
    stream_dims["imag"] = geo_dim   # geo_dim = world_state.hidden_size = 1024
```

`geo_dim` is set to `self.world_state.hidden_size` which is 1024, matching what `imagine_tokens` returns. The `ConditionAssembler` Linear projection for `imag` is thus `Linear(1024, cond_dim)`, handling arbitrary token counts correctly.

### 3. Ablation switches gate construction AND forward/predict

```python
self.use_geo = fw.world_state["enabled"] and fw.world_state["stream"] in ("geo_only", "dual")
self.use_sem = fw.world_state["stream"] in ("sem_only", "dual")
self.use_memory = fw.memory["enabled"]
self.use_imag = fw.imagination["enabled"] and self.use_geo
```

These flags gate submodule construction (WorldStateAdapter, DualMemoryBank, ImaginationAdapter only built when enabled) and gate forward/predict paths via `if self.use_geo`, `if self.use_memory`, `if self.use_imag` checks. The `_assemble` method only passes active streams to ConditionAssembler, and the `stream_dims` dict at construction time only includes active keys — so ConditionAssembler never sees an all-None dict (at least one of geo or sem is always active given `stream` must be one of "geo_only"/"sem_only"/"dual").

## Test Command and Output

```
python -m pytest tests/geomemvla/test_geomemvla_framework.py -v
```

```
tests/geomemvla/test_geomemvla_framework.py::test_framework_is_registered PASSED [ 50%]
tests/geomemvla/test_geomemvla_framework.py::test_default_config_has_ablation_switches PASSED [100%]
======================== 2 passed, 7 warnings in 17.37s ========================
```

Full suite: `python -m pytest tests/geomemvla/ -v` → 12 passed.

## Commit Hash

`65a7122`

## Concerns / Deviations

1. **`_auto_import_framework_modules` resilience**: The brief's test calls `bf._auto_import_framework_modules()` expecting it not to raise. In this environment, `QwenAdapter.py` fails on import (`Qwen3VLForConditionalGeneration` not in installed transformers). `G` (GeoMemoryVLA) comes before `Q` (QwenAdapter) alphabetically so our module IS successfully imported first, but the exception from QwenAdapter propagated and crashed the test. Applied a minimal fix to `base_framework.py`: wrapped each `importlib.import_module` call in try/except with a warning log. This is a defensive fix that matches the intended behavior of the auto-import mechanism (collect all available frameworks, skip broken ones).

2. **Vendored world model forward signature (GPU-unverifiable)**: Cannot confirm `WorldStateAdapter.encode()` and `ImaginationAdapter.imagine_tokens()`/`training_loss()` forward signatures without a GPU and the VGGT model weights. The implementation uses the interfaces exactly as specified in Tasks 1 and 5 (`encode(images[B,F,3,H,W])->GeometryState`, `flatten(state)->[B,F*2*tokens,1024]`, `imagine_tokens(images[B,F,3,H,W], forecast_frames)->[B,L_img,1024]`, `training_loss(images, where)->scalar`). The `where` kwarg is passed as `float(kwargs.get("where", 0.0))` defaulting to 0.0.

3. **`_build_vggt_pixels` vs `_build_image_window` overlap**: Both methods convert PIL images to tensors but serve different purposes — `_build_vggt_pixels` for single-frame world state encoding, `_build_image_window` for multi-frame imagination. These are kept separate as per the brief.

---

## Code Review Fix-up (2026-06-29)

### Finding 1 — Narrow `except Exception` to `except ImportError` in `base_framework.py`

**Before (both call sites):**
```python
except Exception as e:
    logger.warning(f"Could not auto-import framework module {module_name}.{sub_name}: {e}")
```
and
```python
except Exception as e:
    logger.warning(f"Could not auto-import framework module {module_name}: {e}")
```

**After (both call sites):**
```python
except ImportError as e:  # [Geo-MemoryVLA] tolerate peer modules whose heavy deps are absent in this env (e.g. QwenAdapter.py needs Qwen3VLForConditionalGeneration)
    logger.warning(f"Could not auto-import framework module {module_name}.{sub_name}: {e}")
```
and
```python
except ImportError as e:  # [Geo-MemoryVLA] tolerate peer modules whose heavy deps are absent in this env (e.g. QwenAdapter.py needs Qwen3VLForConditionalGeneration)
    logger.warning(f"Could not auto-import framework module {module_name}: {e}")
```

### Finding 2 — Guard against empty `stream_dims` in `GeoMemoryVLA.__init__`

**Before:**
```python
        self.assembler = ConditionAssembler(stream_dims=stream_dims, out_dim=cond_dim)
```

**After:**
```python
        # [Geo-MemoryVLA] ConditionAssembler crashes on an all-None/empty stream set
        # (torch.cat of empty). A contradictory ablation (e.g. world_state.enabled=False
        # with stream="geo_only") would yield no active streams — fail fast with a clear msg.
        if not stream_dims:
            raise ValueError(
                "GeoMemoryVLA: no active condition streams — check framework.world_state.enabled "
                "and framework.world_state.stream (need at least one of geo/sem active)."
            )
        self.assembler = ConditionAssembler(stream_dims=stream_dims, out_dim=cond_dim)
```

### Verification

**AST parse check:**
```
parse OK
```

**test_geomemvla_framework.py -v:**
```
tests/geomemvla/test_geomemvla_framework.py::test_framework_is_registered PASSED [ 50%]
tests/geomemvla/test_geomemvla_framework.py::test_default_config_has_ablation_switches PASSED [100%]
======================== 2 passed, 7 warnings in 16.00s ========================
```

**Full geomemvla suite (tests/geomemvla/ -v):**
```
tests/geomemvla/test_condition_assembler.py::test_projects_and_concats_streams PASSED
tests/geomemvla/test_condition_assembler.py::test_skips_none_streams_for_ablation PASSED
tests/geomemvla/test_dataloader_episode_fields.py::test_sample_dict_includes_episode_and_timestep PASSED
tests/geomemvla/test_geomemvla_framework.py::test_framework_is_registered PASSED
tests/geomemvla/test_geomemvla_framework.py::test_default_config_has_ablation_switches PASSED
tests/geomemvla/test_imagination_adapter.py::test_world_model_loss_contract PASSED
tests/geomemvla/test_imagination_adapter.py::test_adapter_imports PASSED
tests/geomemvla/test_memory_bank.py::test_single_bank_preserves_shape_and_builds_history PASSED
tests/geomemvla/test_memory_bank.py::test_consolidation_caps_memory_length PASSED
tests/geomemvla/test_memory_bank.py::test_dual_bank_handles_none_stream PASSED
tests/geomemvla/test_world_state_adapter.py::test_geometry_state_flatten_contract PASSED
tests/geomemvla/test_world_state_adapter.py::test_adapter_imports_and_exposes_hidden_size PASSED
======================== 12 passed, 7 warnings in 17.60s ========================
```

### Commit Hash

(see below after commit)
