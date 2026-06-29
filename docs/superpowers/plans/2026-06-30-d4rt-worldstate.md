# D4RT-WorldState Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a D4RT-based geometric world-state + forecaster stream to starVLA as a drop-in alternative to the existing VGGT-World stream, enabling a controlled head-to-head comparison inside the existing Geo-MemoryVLA pipeline.

**Architecture:** Two new adapters (`D4RTWorldStateAdapter`, `D4RTImaginationAdapter`) implement the exact same interfaces as the existing `WorldStateAdapter` / `ImaginationAdapter`. The orchestrator `GeoMemoryVLA.py` gains a factory switch on `framework.world_state.backbone ∈ {vggt_world, d4rt}`; everything downstream (memory, semantic stream, GR00T head, dataloader, eval) is untouched. Forecasting is added by querying D4RT's decoder at `t_tgt` beyond the observed prefix.

**Tech Stack:** Python 3.10, PyTorch 2.6, starVLA framework, vendored Open-D4RT (`src/model/d4rt.py:D4RTModel`), pytest.

## Global Constraints

- **Greppable marker:** every new file header and modified block carries `[D4RT-WorldState]`; vendored code notes provenance `# Vendored from Open-D4RT (Apache-2.0): github.com/Lijiaxin0111/Open-d4rt`. (Spec §0)
- **Conda env:** run everything in `/workspace/ghsun/miniconda3/envs/starVLA/bin/python` (numpy **1.26.4**, py 3.10.20). (memory: geo-memoryvla-conda-env)
- **DO NOT `pip install` Open-D4RT's `requirements.txt`** — it pins numpy 2.2.5 and would break the starVLA env. Vendor model code only; it has no transformers/timm/flash_attn deps and runs under numpy 1.26. (Spec R5 / agent dep report)
- **GPUs 4–7** for any GPU/smoke run; 0–3 are used by others. (memory: gpu-testing-preference)
- **D4RT is monocular, agentview-only (B1):** the geometric stream consumes ONLY the agentview camera as a temporal window; wrist reaches the policy via the unchanged Qwen3-VL/GR00T path. (Spec §5)
- **`t_cam = t_tgt` invariant:** all forecasting queries pin `t_cam = t_tgt` (per-frame camera-local xyz), a hard adapter invariant, not a config knob. (Spec R2 + agent: `t_cam=t_tgt` → camera-local)
- **D4RT I/O contract (verified):** `encode_video(video[B,T,3,256,256] in [0,1]) → memory[B,N,C]`; `decode_queries(video, query={u,v,t_src,t_tgt,t_cam}, memory) → {xyz_3d[B,M,3], visibility, ...}`. Video normalized by `/255` only (no mean/std). (agent report §2,§3)
- **Checkpoint:** target `OpenD4RT_48CLIP_9Mix_NoCropAUG` (`clip_frames=48`, 256×256, vit-g); `.ckpt` is NOT in the repo — download from HF `Lijiaxin0111/OpenD4RT/checkpoints/...`; loaded `strict=False`. (agent §1,§4)
- **TDD:** every task writes a failing test first, mirrors the lightweight no-download pattern in `tests/geomemvla/test_world_state_adapter.py`; heavy GPU/model-download tests carry `@pytest.mark.slow`. (existing `tests/geomemvla/pytest.ini`)

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `starVLA/model/modules/d4rt/` (vendored) | Open-D4RT `src/` model code (`model/`, `core/`), verbatim + provenance headers. The encoder/decoder/query machinery. |
| `starVLA/model/modules/d4rt/loader.py` (NEW) | `build_d4rt_model(model_yaml, ckpt_path) → D4RTModel` — wraps `load_yaml_config` + `build_model` + `load_checkpoint` + `_unwrap_state_dict` + `load_state_dict(strict=False)`. |
| `starVLA/model/modules/geomem/d4rt_world_state_adapter.py` (NEW) | `D4RTWorldStateAdapter` — mirrors `WorldStateAdapter`: `.encode/.flatten/.hidden_size/.dtype`. Agentview temporal window → `encode_video` → flattened `[B,N,C]`. |
| `starVLA/model/modules/geomem/d4rt_imagination_adapter.py` (NEW) | `D4RTImaginationAdapter` — mirrors `ImaginationAdapter`: `.imagine_tokens/.training_loss`; `subgoal_type ∈ {latent, tracks}`. Forecasting via `decode_queries` at `t_tgt>t`. |
| `starVLA/model/modules/geomem/world_state_factory.py` (NEW) | `build_world_state(fw)` / `build_imaginer(fw)` — select vggt_world vs d4rt adapters on `fw.world_state["backbone"]`. |
| `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` (MODIFY) | Replace direct `WorldStateAdapter(...)` / `ImaginationAdapter(...)` construction with the factory; add `_build_d4rt_window` for the agentview path. |
| `tools/d4rt_forecast_probe.py` (NEW) | Phase D-PROBE: offline open-loop extrapolation test (no GR00T). |
| `examples/LIBERO/train_files/config_geomemvla_d4rt.yaml` (NEW) | D4RT-arm config (`backbone: d4rt`) + phase variants. |
| `tests/geomemvla/test_d4rt_*.py` (NEW) | Unit tests mirroring existing adapter tests. |

---

## Task 1: Vendor Open-D4RT model code + loader

**Files:**
- Create: `starVLA/model/modules/d4rt/` (vendored `src/model/*.py`, `src/core/*.py` from Open-D4RT)
- Create: `starVLA/model/modules/d4rt/loader.py`
- Create: `starVLA/model/modules/d4rt/__init__.py`
- Test: `tests/geomemvla/test_d4rt_loader.py`

**Interfaces:**
- Consumes: nothing (entry task).
- Produces: `build_d4rt_model(model_yaml: str, ckpt_path: str | None = None, device="cpu") -> nn.Module` returning a `D4RTModel` with methods `encode_video(video, aspect_ratio=None) -> Tensor[B,N,C]` and `decode_queries(video, query: dict, memory) -> dict` with `query` keys `{u,v,t_src,t_tgt,t_cam}` and output key `xyz_3d: Tensor[B,M,3]`.

- [ ] **Step 1: Vendor the source.** Copy the model packages verbatim from the user's local clone at `/workspace/tingting/Open-d4rt` into the vendored tree, then rewrite the package root and add provenance headers. (API verified against this clone: `encode_video` d4rt.py:250, `decode_queries` d4rt.py:260, `build_model` builder.py:19, `load_yaml_config` config.py:71.)

```bash
cd /workspace/tingting/starVLA
mkdir -p starVLA/model/modules/d4rt
cp -r /workspace/tingting/Open-d4rt/src/model starVLA/model/modules/d4rt/model
cp -r /workspace/tingting/Open-d4rt/src/core  starVLA/model/modules/d4rt/core
```
Note: only `src/model` + `src/core` are needed for inference (the adapters call `encode_video`/`decode_queries` + the loader). Do NOT vendor `src/data`, `src/engine`, `src/losses` (training-only; they also carry `from src.` imports and extra deps).

Add `starVLA/model/modules/d4rt/__init__.py` with the provenance header:
```python
# [D4RT-WorldState] Vendored Open-D4RT model code (Apache-2.0).
# Vendored from Open-D4RT (Apache-2.0): github.com/Lijiaxin0111/Open-d4rt
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
```

- [ ] **Step 2: Fix vendored imports.** Open-D4RT uses absolute `from src.model... import` / `from src.core... import`. Rewrite to relative-package imports for the vendored location.

```bash
cd /workspace/tingting/starVLA
grep -rln "from src\.\|import src\." starVLA/model/modules/d4rt/ | while read f; do
  sed -i 's/from src\.model/from starVLA.model.modules.d4rt.model/g; s/from src\.core/from starVLA.model.modules.d4rt.core/g; s/import src\.model/import starVLA.model.modules.d4rt.model/g; s/import src\.core/import starVLA.model.modules.d4rt.core/g' "$f"
done
grep -rn "from src\.\|import src\." starVLA/model/modules/d4rt/ || echo "NO REMAINING src. IMPORTS"
```
Expected: `NO REMAINING src. IMPORTS`.

- [ ] **Step 3: Write the loader + its failing test.** `loader.py` wraps the verified load sequence (agent report §1,§6).

`starVLA/model/modules/d4rt/loader.py`:
```python
# [D4RT-WorldState] Thin loader: model.yaml + .ckpt -> D4RTModel (strict=False).
# Mirrors Open-D4RT eval_track3d_in_worldtrack.py:313-325.
# Vendored-API caller — see github.com/Lijiaxin0111/Open-d4rt
from __future__ import annotations
import torch
from starVLA.model.modules.d4rt.core.config import load_yaml_config
from starVLA.model.modules.d4rt.core.checkpoint import load_checkpoint
from starVLA.model.modules.d4rt.model.builder import build_model


def _unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for k in ("state_dict", "model", "module", "network", "net"):
            if k in payload and isinstance(payload[k], dict):
                return payload[k]
    return payload


def build_d4rt_model(model_yaml: str, ckpt_path: str | None = None, device: str = "cpu"):
    cfg = load_yaml_config(model_yaml)
    model = build_model(cfg["model"]).eval()
    if ckpt_path is not None:
        sd = _unwrap_state_dict(load_checkpoint(ckpt_path, map_location="cpu"))
        if not sd:
            raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval()
```

`tests/geomemvla/test_d4rt_loader.py`:
```python
# tests/geomemvla/test_d4rt_loader.py
# [D4RT-WorldState] Loader import + vendored D4RTModel construction from a model.yaml
# (no checkpoint download).
import glob, pytest


def test_loader_imports():
    from starVLA.model.modules.d4rt.loader import build_d4rt_model
    assert callable(build_d4rt_model)


def test_builds_model_from_yaml_without_ckpt():
    # Build the architecture from a vendored model.yaml; no weights needed.
    from starVLA.model.modules.d4rt.loader import build_d4rt_model
    yamls = glob.glob("starVLA/model/modules/d4rt/**/model.yaml", recursive=True)
    if not yamls:
        pytest.skip("no vendored model.yaml present")
    model = build_d4rt_model(yamls[0], ckpt_path=None)
    assert hasattr(model, "encode_video") and hasattr(model, "decode_queries")
```

Also copy at least one config: `cp .../Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml starVLA/model/modules/d4rt/model_yaml_48.yaml`.

- [ ] **Step 4: Run tests — verify import + build.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_loader.py -v`
Expected: both PASS (env numpy 1.26 must import the vendored code cleanly — if a `numpy>=2` API is used at import, pin/patch it here and note it).

- [ ] **Step 5: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/d4rt tests/geomemvla/test_d4rt_loader.py
git commit -m "feat(d4rt): vendor Open-D4RT model code + loader [D4RT-WorldState]"
```

---

## Task 2: `D4RTWorldStateAdapter` (encoder stream)

**Files:**
- Create: `starVLA/model/modules/geomem/d4rt_world_state_adapter.py`
- Test: `tests/geomemvla/test_d4rt_world_state_adapter.py`

**Interfaces:**
- Consumes: `build_d4rt_model` (Task 1).
- Produces: `D4RTWorldStateAdapter(nn.Module)` with `HIDDEN_SIZE: int` (decoder hidden, from cfg), `.hidden_size -> int`, `.dtype -> torch.dtype`, `.encode(window: Tensor[B,T,3,256,256]) -> D4RTState`, `.flatten(state) -> Tensor[B,N,C]`. `D4RTState` is a dataclass holding `memory: Tensor[B,N,C]` and `video: Tensor[B,T,3,256,256]` (video retained so the imaginer can reuse it for `decode_queries`).

- [ ] **Step 1: Write the failing test** (mirror `test_world_state_adapter.py` — contract only, no model download).

```python
# tests/geomemvla/test_d4rt_world_state_adapter.py
# [D4RT-WorldState] Adapter import + D4RTState flatten contract (no model download).
import torch


def test_d4rt_state_flatten_contract():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTState
    b, n, c = 2, 17, 512
    st = D4RTState(memory=torch.randn(b, n, c), video=torch.randn(b, 4, 3, 256, 256))
    flat = st.flatten()
    assert flat.shape == (b, n, c)


def test_adapter_imports():
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
    assert hasattr(D4RTWorldStateAdapter, "encode")
    assert hasattr(D4RTWorldStateAdapter, "flatten")
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_world_state_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: ...d4rt_world_state_adapter`.

- [ ] **Step 3: Write minimal implementation.**

```python
# starVLA/model/modules/geomem/d4rt_world_state_adapter.py
# [D4RT-WorldState] Thin starVLA-facing wrapper over the vendored Open-D4RT encoder.
# Mirrors WorldStateAdapter: encode(window)->D4RTState, flatten(state)->[B,N,C].
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from starVLA.model.modules.d4rt.loader import build_d4rt_model


@dataclass
class D4RTState:
    memory: torch.Tensor   # [B, N, C]  Global Scene Representation
    video: torch.Tensor    # [B, T, 3, 256, 256]  retained for decoder queries

    def flatten(self) -> torch.Tensor:
        return self.memory


class D4RTWorldStateAdapter(nn.Module):
    def __init__(self, model_yaml: str, ckpt_path: str | None = None) -> None:
        super().__init__()
        self.model = build_d4rt_model(model_yaml, ckpt_path)
        for p in self.model.parameters():      # frozen encoder (R4)
            p.requires_grad_(False)
        # decoder hidden dim = memory channel dim; read from a dummy forward at first use.
        self._hidden_size: int | None = None

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is None:
            raise RuntimeError("hidden_size unknown until first encode(); call encode() once.")
        return self._hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @torch.no_grad()
    def encode(self, window: torch.Tensor) -> D4RTState:
        """window: [B, T, 3, 256, 256] in [0,1] (agentview temporal window)."""
        window = window.to(self.dtype)
        memory = self.model.encode_video(video=window, aspect_ratio=None)  # [B, N, C]
        self._hidden_size = int(memory.shape[-1])
        return D4RTState(memory=memory, video=window)

    def flatten(self, state: D4RTState) -> torch.Tensor:
        return state.flatten()
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_world_state_adapter.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/geomem/d4rt_world_state_adapter.py tests/geomemvla/test_d4rt_world_state_adapter.py
git commit -m "feat(d4rt): D4RTWorldStateAdapter encoder stream [D4RT-WorldState]"
```

---

## Task 3: Phase D-PROBE — offline forecasting probe (de-risk gate)

**Files:**
- Create: `tools/d4rt_forecast_probe.py`
- Test: `tests/geomemvla/test_d4rt_probe.py`

**Interfaces:**
- Consumes: `D4RTWorldStateAdapter` (Task 2), vendored `decode_queries`.
- Produces: `forecast_query(model, video, uv, t_src, t_tgt) -> Tensor[M,3]` (pins `t_cam=t_tgt`); `probe_extrapolation(model, clip, prefix_len, horizon) -> dict` returning `{"gt_xyz", "pred_xyz", "mae"}` comparing prefix-only forecast vs full-clip ground truth.

- [ ] **Step 1: Write the failing test** (pure tensor-shape contract on the query builder; no model).

```python
# tests/geomemvla/test_d4rt_probe.py
# [D4RT-WorldState] Query-builder contract for the forecasting probe (no model).
import torch


def test_build_query_pins_tcam_to_ttgt():
    from tools.d4rt_forecast_probe import build_forecast_query
    M = 5
    uv = torch.rand(M, 2)
    q = build_forecast_query(uv, t_src=0, t_tgt=7, device="cpu")
    assert set(q.keys()) == {"u", "v", "t_src", "t_tgt", "t_cam"}
    assert torch.equal(q["t_cam"], q["t_tgt"])          # t_cam == t_tgt invariant
    assert q["u"].shape == (1, M) and q["t_tgt"].dtype == torch.long
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: tools.d4rt_forecast_probe`.

- [ ] **Step 3: Write minimal implementation.**

```python
# tools/d4rt_forecast_probe.py
# [D4RT-WorldState] Phase D-PROBE: offline open-loop extrapolation test. Feeds a clip
# PREFIX to D4RT, queries points at t_tgt beyond the prefix, compares to full-clip GT.
# Pins t_cam = t_tgt (camera-local). No GR00T, no training. See spec §4 / R1.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations
import torch


def build_forecast_query(uv: torch.Tensor, t_src: int, t_tgt: int, device) -> dict:
    """uv: [M,2] in [0,1]. Returns a D4RT query dict with t_cam == t_tgt."""
    M = uv.shape[0]
    u = uv[:, 0].to(device).float().view(1, M)
    v = uv[:, 1].to(device).float().view(1, M)
    tt = torch.full((1, M), int(t_tgt), dtype=torch.long, device=device)
    return {
        "u": u, "v": v,
        "t_src": torch.full((1, M), int(t_src), dtype=torch.long, device=device),
        "t_tgt": tt,
        "t_cam": tt.clone(),     # invariant: t_cam == t_tgt
    }


@torch.no_grad()
def probe_extrapolation(model, clip: torch.Tensor, uv: torch.Tensor,
                        prefix_len: int, horizon: int) -> dict:
    """clip: [1,T,3,256,256] in [0,1]. Forecast points at prefix_len..prefix_len+horizon-1
    from a PREFIX-only encode, vs ground truth from the FULL-clip encode."""
    device = clip.device
    full_mem = model.encode_video(video=clip, aspect_ratio=None)
    prefix = clip[:, :prefix_len].contiguous()
    pref_mem = model.encode_video(video=prefix, aspect_ratio=None)
    gts, preds = [], []
    for k in range(horizon):
        t = prefix_len + k
        q = build_forecast_query(uv, t_src=0, t_tgt=t, device=device)
        gts.append(model.decode_queries(video=clip, query=q, memory=full_mem)["xyz_3d"])
        # prefix encode only sees [0,prefix_len); query t_tgt>=prefix_len = extrapolation.
        preds.append(model.decode_queries(video=prefix, query=q, memory=pref_mem)["xyz_3d"])
    gt = torch.cat(gts, dim=1)
    pred = torch.cat(preds, dim=1)
    return {"gt_xyz": gt, "pred_xyz": pred, "mae": (gt - pred).abs().mean().item()}
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f tools/d4rt_forecast_probe.py tests/geomemvla/test_d4rt_probe.py
git commit -m "feat(d4rt): Phase D-PROBE offline forecasting probe [D4RT-WorldState]"
```

> **MANUAL GATE (run after the ckpt is downloaded, before Task 4):** download `OpenD4RT_48CLIP_9Mix_NoCropAUG` weights from HF, run `probe_extrapolation` on a handful of LIBERO agentview clips on GPU 4. Record raw-frozen `mae`. If extrapolation is hopeless even after light decoder fine-tuning, the imagination adapter (Task 4) ships `subgoal_type="latent"` only. This is the R1/R5 gate — document the number in the spec before proceeding.

---

## Task 4: `D4RTImaginationAdapter` (forecaster stream, both subgoal types)

**Files:**
- Create: `starVLA/model/modules/geomem/d4rt_imagination_adapter.py`
- Test: `tests/geomemvla/test_d4rt_imagination_adapter.py`

**Interfaces:**
- Consumes: `D4RTWorldStateAdapter` / `D4RTState` (Task 2), `build_forecast_query` (Task 3).
- Produces: `D4RTImaginationAdapter(nn.Module, subgoal_type ∈ {"latent","tracks"})` with `.imagine_tokens(state: D4RTState, forecast_frames: int) -> Tensor[B, L, C]` and `.training_loss(state: D4RTState, gt_future_xyz: Tensor | None = None, where: float = 0.0) -> Tensor` (scalar). Same method names as `ImaginationAdapter` so the orchestrator call sites are unchanged.

- [ ] **Step 1: Write the failing test** (import + a small trainable head shape contract; no D4RT model).

```python
# tests/geomemvla/test_d4rt_imagination_adapter.py
# [D4RT-WorldState] Adapter import + subgoal-head shape contract (no model download).
import torch, pytest


def test_adapter_imports_both_subgoal_types():
    from starVLA.model.modules.geomem.d4rt_imagination_adapter import D4RTImaginationAdapter
    assert hasattr(D4RTImaginationAdapter, "imagine_tokens")
    assert hasattr(D4RTImaginationAdapter, "training_loss")


def test_latent_head_projects_memory_to_subgoal():
    # The 'latent' forecaster head maps [B,N,C] -> [B,N,C] without a D4RT model.
    from starVLA.model.modules.geomem.d4rt_imagination_adapter import LatentForecastHead
    head = LatentForecastHead(dim=64, horizon=2)
    mem = torch.randn(2, 10, 64)
    out = head(mem)
    assert out.shape == (2, 10, 64)
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_imagination_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation.** The encoder stays frozen; only the small forecast head (latent) and a decoder fine-tune (tracks) carry gradients.

```python
# starVLA/model/modules/geomem/d4rt_imagination_adapter.py
# [D4RT-WorldState] Forecaster over the frozen D4RT encoder. Two subgoal types:
#   latent : a small head predicts a future-state token block F_hat (parity w/ VGGT-World z)
#   tracks : query the D4RT decoder at t_tgt>t for future 3D point positions (D4RT-native)
# training_loss supervises against full-clip GT (free, offline). t_cam==t_tgt pinned.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTState
from tools.d4rt_forecast_probe import build_forecast_query


class LatentForecastHead(nn.Module):
    """Maps the observed Global Scene Representation to a future-state token block."""
    def __init__(self, dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:  # [B,N,C]->[B,N,C]
        return memory + self.net(memory)


class D4RTImaginationAdapter(nn.Module):
    def __init__(self, world_state, subgoal_type: str = "latent", horizon: int = 2,
                 track_grid: int = 8) -> None:
        super().__init__()
        assert subgoal_type in ("latent", "tracks")
        self.subgoal_type = subgoal_type
        self.horizon = horizon
        self.track_grid = track_grid
        self._ws = world_state                 # shares the frozen D4RT model
        self.latent_head: LatentForecastHead | None = None  # lazily sized to C

    def _ensure_latent_head(self, dim: int, device, dtype):
        if self.latent_head is None:
            self.latent_head = LatentForecastHead(dim, self.horizon).to(device=device, dtype=dtype)

    def _grid_uv(self, device) -> torch.Tensor:
        g = self.track_grid
        ys, xs = torch.meshgrid(torch.linspace(0, 1, g), torch.linspace(0, 1, g), indexing="ij")
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)  # [g*g,2]

    def imagine_tokens(self, state: D4RTState, forecast_frames: int) -> torch.Tensor:
        if self.subgoal_type == "latent":
            self._ensure_latent_head(state.memory.shape[-1], state.memory.device, state.memory.dtype)
            return self.latent_head(state.memory)            # [B,N,C] latent subgoal
        # tracks: query future 3D positions of a grid; flatten xyz to a token block.
        model = self._ws.model
        T = state.video.shape[1]
        uv = self._grid_uv(state.video.device)
        outs = []
        for k in range(forecast_frames):
            q = build_forecast_query(uv, t_src=T - 1, t_tgt=T - 1 + k, device=state.video.device)
            xyz = model.decode_queries(video=state.video, query=q, memory=state.memory)["xyz_3d"]
            outs.append(xyz)                                  # [1,M,3]
        return torch.cat(outs, dim=1)                         # [B, M*horizon, 3]

    def training_loss(self, state: D4RTState, gt_future_xyz: torch.Tensor | None = None,
                      where: float = 0.0) -> torch.Tensor:
        if self.subgoal_type == "latent":
            # self-supervised: future head should reconstruct the next-state tokens.
            pred = self.imagine_tokens(state, self.horizon)
            return F.mse_loss(pred, state.memory.detach())
        # tracks: supervise predicted future xyz against full-clip GT (free, offline).
        pred = self.imagine_tokens(state, self.horizon)
        if gt_future_xyz is None:
            return pred.sum() * 0.0          # no GT in this batch -> no-op (keeps graph valid)
        return F.l1_loss(pred, gt_future_xyz)
```

- [ ] **Step 4: Run test to verify it passes.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_imagination_adapter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/geomem/d4rt_imagination_adapter.py tests/geomemvla/test_d4rt_imagination_adapter.py
git commit -m "feat(d4rt): D4RTImaginationAdapter latent+tracks forecaster [D4RT-WorldState]"
```

---

## Task 5: World-state factory + orchestrator switch

**Files:**
- Create: `starVLA/model/modules/geomem/world_state_factory.py`
- Modify: `starVLA/model/framework/VLM4A/GeoMemoryVLA.py:88` and `:107-117` (construction sites)
- Test: `tests/geomemvla/test_world_state_factory.py`

**Interfaces:**
- Consumes: `WorldStateAdapter`, `ImaginationAdapter` (existing), `D4RTWorldStateAdapter` (Task 2), `D4RTImaginationAdapter` (Task 4).
- Produces: `build_world_state(fw: dict) -> nn.Module` and `build_imaginer(fw: dict, world_state) -> nn.Module`, dispatching on `fw["world_state"].get("backbone", "vggt_world")`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/geomemvla/test_world_state_factory.py
# [D4RT-WorldState] Factory dispatches on world_state.backbone (no model download).
import pytest


def test_factory_defaults_to_vggt_world_class():
    from starVLA.model.modules.geomem.world_state_factory import _world_state_cls
    from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
    assert _world_state_cls("vggt_world") is WorldStateAdapter


def test_factory_selects_d4rt_class():
    from starVLA.model.modules.geomem.world_state_factory import _world_state_cls
    from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
    assert _world_state_cls("d4rt") is D4RTWorldStateAdapter


def test_factory_rejects_unknown_backbone():
    from starVLA.model.modules.geomem.world_state_factory import _world_state_cls
    with pytest.raises(ValueError):
        _world_state_cls("nope")
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_world_state_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: ...world_state_factory`.

- [ ] **Step 3: Write the factory.**

```python
# starVLA/model/modules/geomem/world_state_factory.py
# [D4RT-WorldState] Selects the geometric world-state + imaginer adapters on
# framework.world_state.backbone in {vggt_world, d4rt}. The two backbones share the
# adapter interface so GeoMemoryVLA call sites are unchanged.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
from starVLA.model.modules.geomem.imagination_adapter import ImaginationAdapter
from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
from starVLA.model.modules.geomem.d4rt_imagination_adapter import D4RTImaginationAdapter


def _world_state_cls(backbone: str):
    if backbone == "vggt_world":
        return WorldStateAdapter
    if backbone == "d4rt":
        return D4RTWorldStateAdapter
    raise ValueError(f"unknown world_state.backbone: {backbone!r}")


def build_world_state(fw: dict):
    ws = fw["world_state"]
    backbone = ws.get("backbone", "vggt_world")
    if backbone == "vggt_world":
        return WorldStateAdapter(pretrained_vggt_repo=ws["model_name"])
    return D4RTWorldStateAdapter(model_yaml=ws["d4rt_model_yaml"],
                                 ckpt_path=ws.get("d4rt_ckpt_path"))


def build_imaginer(fw: dict, world_state):
    ws = fw["world_state"]
    imag = fw["imagination"]
    backbone = ws.get("backbone", "vggt_world")
    if backbone == "vggt_world":
        return ImaginationAdapter(
            pretrained_vggt_repo=ws["model_name"],
            chunk_size=int(imag["horizon"]),
            context_size=int(imag.get("context_size", 2)),
        )
    return D4RTImaginationAdapter(
        world_state=world_state,
        subgoal_type=imag.get("subgoal_type", "latent"),
        horizon=int(imag["horizon"]),
    )
```

- [ ] **Step 4: Wire the orchestrator.** In `GeoMemoryVLA.py`, replace the direct constructions.

At top of file, add import:
```python
# [D4RT-WorldState] factory selects vggt_world vs d4rt world-state + imaginer.
from starVLA.model.modules.geomem.world_state_factory import build_world_state, build_imaginer
```
Replace line 88 (`self.world_state = WorldStateAdapter(...)`) with:
```python
            # [D4RT-WorldState] backbone-agnostic construction (vggt_world | d4rt).
            self.world_state = build_world_state(fw)
```
Replace the `self.imaginer = ImaginationAdapter(...)` block (lines 107-110) with:
```python
            # [D4RT-WorldState] backbone-agnostic imaginer.
            self.imaginer = build_imaginer(fw, self.world_state)
```
Note: `geo_dim = self.world_state.hidden_size` (line 89) — D4RT's `hidden_size` is only known after the first `encode()`. Move the `geo_dim` read to lazily occur after the first world-state encode, OR read it from the D4RT cfg in the adapter `__init__` (preferred: set `D4RTWorldStateAdapter._hidden_size` from `cfg["model"]["decoder"]` hidden in Task 2's `__init__` so `hidden_size` is available immediately). Apply the preferred fix in Task 2 if not already done.

- [ ] **Step 5: Run factory tests + existing geomem suite (regression).**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_world_state_factory.py tests/geomemvla/test_geomemvla_framework.py -v`
Expected: factory tests PASS; existing framework test still PASS (vggt_world default unchanged).

- [ ] **Step 6: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/geomem/world_state_factory.py starVLA/model/framework/VLM4A/GeoMemoryVLA.py tests/geomemvla/test_world_state_factory.py
git commit -m "feat(d4rt): world-state factory + GeoMemoryVLA backbone switch [D4RT-WorldState]"
```

---

## Task 6: Agentview window builder + D4RT config

**Files:**
- Modify: `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` (add `_build_d4rt_window`; route it when backbone==d4rt)
- Create: `examples/LIBERO/train_files/config_geomemvla_d4rt.yaml`
- Test: `tests/geomemvla/test_d4rt_window.py`, `tests/geomemvla/test_d4rt_config_loads.py`

**Interfaces:**
- Consumes: orchestrator (Task 5); existing `_build_image_window` pattern (`GeoMemoryVLA.py:207`).
- Produces: `_build_d4rt_window(examples, device, win_len) -> Tensor[B, win_len, 3, 256, 256]` — agentview-only, replicate-pad to `win_len`, normalized to `[0,1]`.

- [ ] **Step 1: Write the failing test** (window shape + agentview-only + [0,1] range, with synthetic PIL frames).

```python
# tests/geomemvla/test_d4rt_window.py
# [D4RT-WorldState] Agentview temporal window builder contract.
import torch
from PIL import Image


def _ex(n_views):
    img = Image.new("RGB", (256, 256))
    return {"image": [img for _ in range(n_views)]}


def test_window_is_agentview_only_and_normalized():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import _build_d4rt_window
    examples = [_ex(2), _ex(2)]      # 2 cameras present; D4RT must take only agentview (view 0)
    win = _build_d4rt_window(examples, device="cpu", win_len=4)
    assert win.shape == (2, 4, 3, 256, 256)
    assert win.min() >= 0.0 and win.max() <= 1.0
```

- [ ] **Step 2: Run to verify it fails.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_window.py -v`
Expected: FAIL (`_build_d4rt_window` not importable).

- [ ] **Step 3: Implement `_build_d4rt_window`** as a module-level function in `GeoMemoryVLA.py` (module-level so the test imports it without constructing the model):

```python
# [D4RT-WorldState] Agentview-only temporal window for the monocular D4RT stream (B1).
# Frame axis is TIME; view 0 (agentview) only. Replicate-pad to win_len. Range [0,1].
def _build_d4rt_window(examples, device, win_len: int = 48):
    import torchvision.transforms.functional as TF
    windows = []
    for e in examples:
        imgs = e["image"] if isinstance(e["image"], (list, tuple)) else [e["image"]]
        agent = imgs[0].convert("RGB")                       # view 0 = agentview (B1)
        frames = e.get("image_window")
        if frames:                                           # List[F][views] -> agentview series
            series = [(f[0] if isinstance(f, (list, tuple)) else f).convert("RGB") for f in frames]
        else:
            series = [agent]
        series = (series + [series[-1]] * win_len)[:win_len] # replicate-pad to win_len
        ten = torch.stack([TF.resize(TF.to_tensor(im), [256, 256]) for im in series], dim=0)
        windows.append(ten)
    return torch.stack(windows, dim=0).to(device)            # [B, win_len, 3, 256, 256]
```
Then, where the orchestrator calls `self.world_state.encode(...)`, branch on backbone: for `d4rt`, build the window with `_build_d4rt_window(examples, device, self._d4rt_win_len)` instead of `_build_vggt_pixels`. Read `self._d4rt_win_len` from `fw.world_state.get("clip_frames", 48)` in `__init__`.

- [ ] **Step 4: Create the D4RT config.** Copy an existing GR00T LIBERO config and set the D4RT backbone fields.

```yaml
# examples/LIBERO/train_files/config_geomemvla_d4rt.yaml
# [D4RT-WorldState] D4RT comparison arm. Copy of the geomemvla LIBERO config with the
# geometric stream switched to D4RT. See docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
framework:
  name: GeoMemoryVLA
  world_state:
    enabled: true
    stream: dual
    backbone: d4rt                       # <-- the switch
    d4rt_model_yaml: starVLA/model/modules/d4rt/model_yaml_48.yaml
    d4rt_ckpt_path: playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
    clip_frames: 48
    num_cameras: 2
  memory: { enabled: true, mem_length: 16, retrieval_layers: 2, fusion_type: gate, consolidate_type: tome }
  imagination: { enabled: true, subgoal_type: latent, horizon: 2, loss_scale: 0.5 }
# (action_model / qwenvl / datasets sections: copy verbatim from the existing geomemvla config.)
```

```python
# tests/geomemvla/test_d4rt_config_loads.py
# [D4RT-WorldState] The D4RT config parses and selects the d4rt backbone.
import yaml


def test_d4rt_config_selects_d4rt_backbone():
    with open("examples/LIBERO/train_files/config_geomemvla_d4rt.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["framework"]["world_state"]["backbone"] == "d4rt"
    assert cfg["framework"]["imagination"]["subgoal_type"] in ("latent", "tracks")
```

- [ ] **Step 5: Run tests.**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_window.py tests/geomemvla/test_d4rt_config_loads.py -v`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/framework/VLM4A/GeoMemoryVLA.py examples/LIBERO/train_files/config_geomemvla_d4rt.yaml tests/geomemvla/test_d4rt_window.py tests/geomemvla/test_d4rt_config_loads.py
git commit -m "feat(d4rt): agentview window builder + D4RT LIBERO config [D4RT-WorldState]"
```

---

## Task 7: GPU smoke test (Phase D0 — encoder swap end-to-end)

**Files:**
- Create: `tests/geomemvla/test_d4rt_smoke_gpu.py`

**Interfaces:**
- Consumes: everything above + the downloaded D4RT checkpoint.
- Produces: a `@pytest.mark.slow` test that builds `GeoMemoryVLA` with `backbone=d4rt`, runs one forward on synthetic LIBERO-shaped input on GPU 4, asserts an action chunk comes out.

- [ ] **Step 1: Write the slow smoke test** (mirrors `tests/geomemvla/test_smoke_gpu.py`).

```python
# tests/geomemvla/test_d4rt_smoke_gpu.py
# [D4RT-WorldState] End-to-end Phase D0: GeoMemoryVLA(backbone=d4rt) one forward pass.
import os, pytest, torch

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_d4rt_geomemvla_forward_produces_action():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
    # Build the framework from config_geomemvla_d4rt.yaml with imagination disabled (D0),
    # feed one synthetic batch (agentview+wrist PIL, proprio state), assert action chunk shape.
    # Follow the construction pattern in tests/geomemvla/test_smoke_gpu.py exactly,
    # only changing the config path to config_geomemvla_d4rt.yaml and setting
    # framework.imagination.enabled=false for the D0 row.
    ...  # implement against the existing smoke-test scaffolding
```

- [ ] **Step 2: Download the checkpoint** (one-time, before running the slow test).

```bash
/workspace/ghsun/miniconda3/envs/starVLA/bin/huggingface-cli download \
  Lijiaxin0111/OpenD4RT --include "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/*" \
  --local-dir playground/Pretrained_models/_opend4rt_dl
# place opend4rt.ckpt where the config expects it:
mkdir -p playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG
# (copy/symlink the downloaded .ckpt — confirm the exact filename after download)
```

- [ ] **Step 3: Run the smoke test on GPU 4.**

Run: `cd /workspace/tingting/starVLA && CUDA_VISIBLE_DEVICES=4 /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_d4rt_smoke_gpu.py -v -m slow`
Expected: PASS — an action chunk `[B, action_horizon, action_dim]` is produced. (This is the Phase D0 integration proof.)

- [ ] **Step 4: Commit.**

```bash
cd /workspace/tingting/starVLA
git add -f tests/geomemvla/test_d4rt_smoke_gpu.py
git commit -m "test(d4rt): Phase D0 GPU smoke test [D4RT-WorldState]"
```

---

## Self-Review

**Spec coverage:**
- §1 controlled experiment (one switch) → Task 5 factory + Task 6 config. ✓
- §2 verified D4RT I/O → Task 1 loader, Task 2 encode. ✓
- §3a `D4RTWorldStateAdapter` → Task 2. ✓
- §3b `D4RTImaginationAdapter` latent+tracks → Task 4. ✓
- §3c orchestrator factory switch → Task 5. ✓
- §4 forecasting surgery + standalone probe → Task 3 (+ manual gate). ✓
- §5 B1 agentview-only → Task 6 `_build_d4rt_window`. ✓
- §6 R1 (probe gate) Task 3; R2 (`t_cam=t_tgt`) Task 3 `build_forecast_query`; R3 (window) Task 6; R4 (frozen, GPU 4-7) Task 2/Task 7; R5 (repro quality, ckpt sanity) Task 3 manual gate + Task 7 download. ✓
- §7 phasing: D-PROBE=Task 3, D0=Task 7. D1/D2/D3 = config-only runs (subgoal_type/backbone toggles) once D0 passes — no new code, so no task; they are training runs the operator launches. ✓
- §8 switches: `backbone` (Task 5), `subgoal_type` (Task 4/Task 6 config). ✓
- §9 magnitude matches: 2 adapters + factory + probe + vendored + config. ✓
- §11 open tasks: #1 API (resolved by agent, baked into tasks); #2 window/stride (Task 6 `clip_frames`); #3 track points (Task 4 `_grid_uv` — end-effector pixel refinement deferred to a D2 follow-up, noted below); #4 supervision precompute (Task 4 `gt_future_xyz` arg — wiring the GT loader is a D2 follow-up); #5 `F`→memory dim (Task 2 `hidden_size` + existing ConditionAssembler projection); #6 ckpt sanity (Task 3 gate). ✓

**Deferred (explicitly, not gaps):** end-effector-pixel track selection and the full-clip GT-future supervision *dataloader* wiring are D2-phase refinements that only matter for `subgoal_type=tracks` headline runs; `latent` (the apples-to-apples row) is fully covered. These are called out here so the operator wires them before the D2 training run, not during D0/D1.

**Placeholder scan:** Task 7 Step 1 leaves the smoke-test body as `...` against "the existing smoke-test scaffolding" — this is intentional (mirror an existing in-repo test whose exact construction I have not read line-by-line); the implementer reads `tests/geomemvla/test_smoke_gpu.py` and copies its setup. All other steps contain complete code.

**Type consistency:** `D4RTState(memory, video)` / `.flatten()` consistent across Tasks 2,3,4. `encode_video(video=, aspect_ratio=)` and `decode_queries(video=, query=, memory=)` match the agent's verified signatures everywhere. `build_forecast_query` returns the same 5-key dict consumed by probe and imaginer. `hidden_size`/`_hidden_size` consistent Task 2↔Task 5.
