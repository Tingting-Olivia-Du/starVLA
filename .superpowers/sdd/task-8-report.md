# Task 8 Report — geo_only GPU Smoke Test

**Status:** DONE_WITH_CONCERNS  
**Commit:** `dedcc87`  
**Date:** 2026-06-29

---

## 1. Design Gap Fixed: Qwen3-VL Construction Was Unconditional

**Before (lines 69-73 in GeoMemoryVLA.py):**
```python
self.qwen_vl_interface = get_vlm_model(config=self.config)
sem_dim = int(self.qwen_vl_interface.model.config.hidden_size)

self.use_geo = fw.world_state["enabled"] and fw.world_state["stream"] in ("geo_only", "dual")
self.use_sem = fw.world_state["stream"] in ("sem_only", "dual")
```
`get_vlm_model()` was called unconditionally **before** `use_sem` was even computed.
For `stream="geo_only"` this triggered `Qwen3VLForConditionalGeneration` import,
which fails in this env (transformers 4.53.2).

**After:**
```python
# [Geo-MemoryVLA] Determine which streams are active BEFORE constructing Qwen3-VL,
# so that geo_only ablations can run in environments where Qwen3VLForConditionalGeneration
# is absent (e.g. transformers < 4.54). Qwen3-VL is only constructed when use_sem=True.
self.use_geo = fw.world_state["enabled"] and fw.world_state["stream"] in ("geo_only", "dual")
self.use_sem = fw.world_state["stream"] in ("sem_only", "dual")

if self.use_sem:
    self.qwen_vl_interface = get_vlm_model(config=self.config)
    sem_dim = int(self.qwen_vl_interface.model.config.hidden_size)
else:
    # [Geo-MemoryVLA] geo_only ablation: no VLM. Read sem_dim from config as a
    # fallback so ConditionAssembler / DualMemoryBank dim wiring still works if
    # sem streams are later enabled without changing this branch.
    sem_dim = int(fw.qwenvl.get("vl_hidden_dim", 2048))
```

---

## 2. _encode Fix: geo_only Path (No VLM Forward)

**Before:**
```python
def _encode(self, examples):
    images = [e["image"] for e in examples]
    instructions = [e["lang"] for e in examples]
    qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(...)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
    sem = out.hidden_states[-1] if self.use_sem else None  # [B, L, H]
    geo = None
    if self.use_geo:
        pix = self._build_vggt_pixels(images, device=out.hidden_states[-1].device)
        ...
```
Even when `use_sem=False`, it dereferenced `out.hidden_states[-1]` for the VGGT device.

**After:**
```python
# [Geo-MemoryVLA] geo_only ablation: skip VLM forward entirely when use_sem=False
if self.use_sem:
    instructions = [e["lang"] for e in examples]
    qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(...)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
    sem = out.hidden_states[-1]  # [B, L, H]
    vggt_device = sem.device
else:
    sem = None
    # [Geo-MemoryVLA] Infer device from the VGGT backbone when VLM is absent.
    vggt_device = next(self.world_state.parameters()).device

geo = None
if self.use_geo:
    pix = self._build_vggt_pixels(images, device=vggt_device)
    ...
```

---

## 3. Smoke Test Code

File: `tests/geomemvla/test_smoke_gpu.py`

```python
# [Geo-MemoryVLA] End-to-end GPU smoke test: geo_only stream (no Qwen3-VL).
@pytest.mark.slow
def test_forward_and_predict_geo_only():
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework

    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    cfg.framework.world_state.stream = "geo_only"
    cfg.framework.memory.enabled = False
    cfg.framework.imagination.enabled = False

    model = build_framework(cfg).cuda()
    assert not hasattr(model, "qwen_vl_interface")

    # 2 fake samples, 2 views each (224x224)
    ...
    out = model([sample_a, sample_b])
    assert torch.isfinite(out["action_loss"])

    pred = model.predict_action([sample_a])
    assert pred["normalized_actions"].shape == (1, 8, 7)
```

---

## 4. Smoke Run Command + Output

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/tingting/starVLA \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  python -m pytest tests/geomemvla/test_smoke_gpu.py -v -m slow -s
```

**Key output:**
```
[smoke] action_loss = 1.377856
[smoke] predict_action shape = (1, 8, 7)  — PASS
=================== 1 passed in 76.38s ===================
```

---

## 5. Fast Suite Re-Run

```
CUDA_VISIBLE_DEVICES=1 python -m pytest tests/geomemvla/ -v -m "not slow"
```
Result: **13 passed, 1 deselected** — unchanged.

---

## 6. Commit

Hash: `dedcc87`  
Message: `[Geo-MemoryVLA] Add geo_only GPU smoke test; enable Qwen-free ablation`

Files changed:
- `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` — Qwen-free ablation fix (2 locations)
- `tests/geomemvla/test_smoke_gpu.py` — new slow smoke test

---

## 7. Concerns

1. **`predict_action` calls `to_pil_preserve` on images that are already PIL.** This is a no-op (PIL is returned as-is per the docstring), but it adds a small overhead and is semantically redundant when the caller already passes PIL images (as in eval).

2. **`stream="geo_only"` with `world_state.enabled=False`** would raise the `ValueError` ("no active condition streams") rather than silently building a sem-only model. This is intentional fail-fast behavior but could surprise users. The error message is clear.

3. **The `get_vlm_model` import at the top of GeoMemoryVLA.py is still present** even in geo_only mode — it's a module-level import, not a class-level one, so as long as the `QWen3.py` module itself doesn't fail on import (it only fails when you try to instantiate Qwen3VLForConditionalGeneration), this is fine. Confirmed: the auto-importer logs a warning about QwenAdapter but GeoMemoryVLA.py itself imports fine.

4. **geo_only + memory=True** is not tested (DualMemoryBank would receive `sem=None`, which `test_memory_bank.py::test_dual_bank_handles_none_stream` already covers for the bank, but end-to-end with VGGT on GPU is not exercised). Low priority until memory is needed for an experiment.

---

## Final-review fixes

**Date:** 2026-06-29  
**Commit:** (see below)

### Edit 1 — YAML: `imagination.enabled` default changed to `false` with Phase-C comment

**File:** `examples/LIBERO/train_files/geo_memoryvla_libero.yaml`

Before:
```yaml
  imagination:
    enabled: true
    horizon: 4
```

After:
```yaml
  imagination:
    # [Geo-MemoryVLA] OFF by default: the vendored VGGTWorldModel needs multi-frame
    # image windows (>= context_size+chunk_size frames). The LIBERO dataloader yields
    # single frames today, so enabling this requires the Phase-C "image_window" dataloader
    # work (see docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md §7). Flipping
    # this to true WITHOUT that work will fail/degenerate.
    enabled: false
    horizon: 4
```

### Edit 2 — GeoMemoryVLA.py: `import warnings` added + warn-once guard in `_build_image_window`

**File:** `starVLA/model/framework/VLM4A/GeoMemoryVLA.py`

Before (imports):
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
```

After (imports):
```python
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional
```

Before (`_build_image_window` loop body):
```python
        for e in examples:
            frames = e.get("image_window", [e["image"]])
            frame_tensors = []
```

After:
```python
        for e in examples:
            if "image_window" not in e:
                # [Geo-MemoryVLA] Phase-C guard: imagination needs multi-frame windows. If the
                # dataloader did not supply "image_window", we degenerate to the single current
                # frame — warn once so this can't pass silently (see spec §7 Phase-C follow-up).
                if not getattr(self, "_warned_single_frame_window", False):
                    warnings.warn(
                        "GeoMemoryVLA: imagination enabled but no 'image_window' in batch; "
                        "degenerating to single-frame window. Training may fail (vendored "
                        "VGGTWorldModel requires >= context_size+chunk_size frames). See spec §7.",
                        stacklevel=2,
                    )
                    self._warned_single_frame_window = True
            frames = e.get("image_window", [e["image"]])
            frame_tensors = []
```

### Edit 3 — test_smoke_gpu.py: GPU comment updated to reference GPUs 4-7

**File:** `tests/geomemvla/test_smoke_gpu.py`

Before:
```
# correct predict_action output shape. Marked @pytest.mark.slow; needs GPU 1.
#
# Run:
#   cd /workspace/tingting/starVLA
#   CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/tingting/starVLA \
```

After:
```
# correct predict_action output shape. Marked @pytest.mark.slow; needs a free GPU.
# Project convention: use GPUs 4-7 for test/smoke runs (0-3 reserved for others).
#
# Run:
#   cd /workspace/tingting/starVLA
#   CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/workspace/tingting/starVLA \
```

### Verification outputs

1. Fast test suite (13 tests, not slow):
```
================ 13 passed, 1 deselected, 7 warnings in 17.87s =================
```

2. Python parse check:
```
parse OK
```

3. OmegaConf imagination.enabled check:
```
imagination.enabled = False
```

### Commit hash

`b45bf5f` — `[Geo-MemoryVLA] Final review: default imagination off + Phase-C warn-once guard; GPU-convention comment`
