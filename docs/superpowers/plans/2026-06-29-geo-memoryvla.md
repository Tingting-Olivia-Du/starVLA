# Geo-MemoryVLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new starVLA framework `GeoMemoryVLA` whose memory and imagination operate over a dual latent space — VGGT 3D-geometry latent + Qwen3-VL 2D-semantic latent — feeding the existing GR00T flow-matching action head.

**Architecture:** A frozen VGGT-1B produces a 3D world-state latent `z_t`; Qwen3-VL produces semantic/language tokens. A dual memory bank (lifted from MemoryVLA, MIT) stores and retrieves history in both spaces; a flow-transformer imaginer (reimplemented from VGGT-World) predicts a future `z` subgoal. All streams are projected and concatenated into a single `[B, L, H]` condition consumed by the unchanged GR00T head.

**Tech Stack:** Python, PyTorch, starVLA framework, `vggt` package (v0.0.1, already installed), Qwen3-VL (transformers), pytest.

## Global Constraints

- **Coding marker:** Every new file gets a header `# [Geo-MemoryVLA] <role>` + a pointer to `docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md`. Every modified block in an existing file gets an inline `# [Geo-MemoryVLA] <what/why>` comment. Lifted code names provenance (`# Adapted from MemoryVLA (MIT): vla/memory_vla.py:CogMemBank`).
- **Host framework is starVLA** (`/workspace/tingting/starVLA`); all paths below are relative to it unless absolute.
- **Reuse, don't fork:** GR00T head (`modules/action_model/GR00T_ActionHeader.py`), Qwen3-VL wrapper (`modules/vlm/QWen3.py`), and the training loop (`training/train_starvla.py`) are used **unchanged**.
- **GR00T head contract (verified):** `FlowmatchingActionHead.forward(vl_embs[B,L,H], actions[B,T,A], state[B,S]=None, encoder_attention_mask[B,L]=None) -> scalar loss`; `predict_action(vl_embs[B,L,H], state=None, encoder_attention_mask=None) -> [B,T,A]`. `vl_embs` is the **full sequence**, not pooled. `diffusion_model_cfg.cross_attention_dim` MUST be set to the condition hidden size **before** `get_action_model(config)` is called.
- **Framework auto-discovery (verified):** dropping a `.py` in `model/framework/VLM4A/` with `@FRAMEWORK_REGISTRY.register("GeoMemoryVLA")` is sufficient; no import-list edits.
- **VGGT (verified):** `from vggt.models.vggt import VGGT; VGGT.from_pretrained("facebook/VGGT-1B")`; aggregator latent dim `d=1024`; use layer-4, drop the first 5 special tokens.
- **Frozen modules:** VGGT is always frozen (`requires_grad_(False)`, forward under `torch.no_grad()`). Qwen3-VL freezing follows the config (`trainer.freeze_modules`).
- **GPUs:** use 4,5,6,7 for smoke/test runs (0-3 used by others).
- **Test command base:** `cd /workspace/tingting/starVLA && python -m pytest <path> -v`. Tests that need a GPU/model download are marked `@pytest.mark.slow` and excluded from the default fast suite.

---

## File Structure

| File | Responsibility | Status |
| --- | --- | --- |
| `starVLA/model/modules/vggt_world/*` | VGGT-World backbone + flow transformer + losses + solver (11 files) | VENDORED verbatim |
| `starVLA/model/modules/geomem/world_state_adapter.py` | Wrap `FrozenVGGTBackbone` → `GeometryState` → flattened `[B, L_g, 1024]` | NEW (thin adapter) |
| `starVLA/model/modules/geomem/imagination_adapter.py` | Wrap `VGGTWorldModel`: imagination loss + imagined subgoal tokens | NEW (thin adapter) |
| `starVLA/model/modules/geomem/condition_assembler.py` | Project + concat active streams → `[B, L, H]` + mask | NEW |
| `starVLA/model/modules/memory/dual_memory_bank.py` | Geo + semantic memory banks: store / retrieve / ToMe-consolidate | NEW (lifts MemoryVLA) |
| `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` | Orchestrator: VGGT + Qwen3-VL + memory + imaginer → fused `[B,L,H]` → GR00T head | NEW |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Add `episode_id` + `timestep` to sample dict | MODIFY (~2 lines) |
| `examples/LIBERO/train_files/geo_memoryvla_libero.yaml` | Training config with ablation switches | NEW |
| `tests/geomemvla/` | Unit tests for each module | NEW |

**Phasing (each phase is independently runnable):**
- **Phase A** (Tasks 1-2, 6-8 with memory/imagination disabled): vendored VGGT world-state + GR00T. Proves VGGT plugs into the head.
- **Phase B** (Tasks 3-4): dual memory bank + dataloader episode/timestep.
- **Phase C** (Task 5): imagination adapter over the vendored VGGTWorldModel.
- Tasks 6-9 (framework, config, smoke, marker) gate each phase via config switches.

---

## Task 1: Vendor VGGT-World + world-state adapter

**What changed (read this):** an earlier draft hand-wrote a `VGGTWorldStateEncoder` and a toy
imaginer from the paper *abstract*. The real VGGT-World implementation is available at
`/workspace/tingting/vggt-world/vggt/world_model/` (1577 LOC, Meta research license, allows
derivative works + research use, requires acknowledgement in publications). We **vendor it
verbatim** and write only a thin adapter. This task does the vendoring + the world-state
adapter; Task 5 does the imagination adapter.

**Files:**
- Create: `starVLA/model/modules/vggt_world/` ← copied verbatim from
  `/workspace/tingting/vggt-world/vggt/world_model/` (all 12 `.py` files: `__init__`,
  `backbone`, `blocks`, `flow_model`, `losses`, `metrics`, `model`, `rope3d`, `scheduler`,
  `solver`, `state`, `time_embed`).
- Create: `starVLA/model/modules/vggt_world/PROVENANCE.md` (source path, license, commit note).
- Create: `starVLA/model/modules/geomem/__init__.py` (empty)
- Create: `starVLA/model/modules/geomem/world_state_adapter.py`
- Test: `tests/geomemvla/test_world_state_adapter.py`

**Interfaces:**
- Consumes: the vendored `FrozenVGGTBackbone.encode_states(images[B,F,3,H,W]) -> GeometryState`
  (`backbone.py:84`) and `GeometryState.flatten_streams() -> [B, F*2*tokens, 1024]` (`state.py:55`).
- Produces:
  - `class WorldStateAdapter(nn.Module)`:
    - `__init__(self, pretrained_vggt_repo="facebook/VGGT-1B")` — builds `FrozenVGGTBackbone`.
    - `encode(self, images: torch.Tensor) -> GeometryState` — passthrough to backbone.
    - `flatten(self, state) -> torch.Tensor` — `state.flatten_streams()`.
    - property `hidden_size -> int` (= 1024).

- [ ] **Step 1: Vendor the package**

```bash
cd /workspace/tingting/starVLA
mkdir -p starVLA/model/modules/vggt_world tests/geomemvla
touch tests/geomemvla/__init__.py
cp /workspace/tingting/vggt-world/vggt/world_model/*.py starVLA/model/modules/vggt_world/
```

Prepend the `[Geo-MemoryVLA]` provenance header to each vendored file (a one-line sed is
acceptable since these are verbatim copies):

```bash
cd /workspace/tingting/starVLA/starVLA/model/modules/vggt_world
for f in *.py; do
  printf '# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).\n# Source: /workspace/tingting/vggt-world/vggt/world_model/%s\n# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md\n' "$f" | cat - "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
```

Write `PROVENANCE.md`:

```markdown
# [Geo-MemoryVLA] vggt_world provenance
Vendored verbatim from https://github.com/sisyphm/vggt-world
(`vggt/world_model/`), local clone `/workspace/tingting/vggt-world`.
License: Meta research license (LICENSE.txt in the source repo) — permits derivative
works and research use; **publications must acknowledge use**. The original `facebook/VGGT-1B`
weights are non-commercial (use `VGGT-1B-Commercial` for commercial use).
Do not hand-edit these files except for the provenance header; treat as upstream.
```

- [ ] **Step 2: Write the failing test**

```python
# tests/geomemvla/test_world_state_adapter.py
# [Geo-MemoryVLA] Adapter import + GeometryState flatten contract (no model download).
import torch


def test_geometry_state_flatten_contract():
    # Exercises the vendored GeometryState shape contract without loading VGGT.
    from starVLA.model.modules.vggt_world.state import GeometryState

    b, f, tok, d = 2, 3, 7, 1024
    st = GeometryState(
        frame_tokens=torch.randn(b, f, tok, d),
        global_tokens=torch.randn(b, f, tok, d),
        patch_start_idx=5,
        patch_grid=(1, 1),
    )
    flat = st.flatten_streams()
    assert flat.shape == (b, f * 2 * tok, d)


def test_adapter_imports_and_exposes_hidden_size():
    from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
    # Class is importable and declares the dim without instantiating VGGT.
    assert WorldStateAdapter.HIDDEN_SIZE == 1024
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_world_state_adapter.py -v`
Expected: FAIL — `world_state_adapter` not found (the vendored `state.py` test may already pass).

- [ ] **Step 4: Write the adapter**

```python
# starVLA/model/modules/geomem/world_state_adapter.py
# [Geo-MemoryVLA] Thin starVLA-facing wrapper over the vendored VGGT-World backbone.
# Turns images into a GeometryState world state and flattens it for downstream
# memory / condition assembly. Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.vggt_world.backbone import FrozenVGGTBackbone


class WorldStateAdapter(nn.Module):
    HIDDEN_SIZE = 1024

    def __init__(self, pretrained_vggt_repo: str = "facebook/VGGT-1B") -> None:
        super().__init__()
        self.backbone = FrozenVGGTBackbone.from_pretrained(pretrained_vggt_repo)

    @property
    def hidden_size(self) -> int:
        return self.HIDDEN_SIZE

    def encode(self, images: torch.Tensor):
        """images: [B, num_views(*frames), 3, H, W] -> GeometryState (frozen)."""
        return self.backbone.encode_states(images)

    def flatten(self, state) -> torch.Tensor:
        """GeometryState -> [B, F*2*tokens, 1024]."""
        return state.flatten_streams()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_world_state_adapter.py -v`
Expected: PASS (both tests).

> The real `encode()` path (downloads VGGT-1B) is exercised in the Task 8 GPU smoke test.

- [ ] **Step 6: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/vggt_world/ starVLA/model/modules/geomem/ tests/geomemvla/test_world_state_adapter.py tests/geomemvla/__init__.py
git commit -m "[Geo-MemoryVLA] Vendor VGGT-World + add world-state adapter"
```

---

## Task 2: Stream projector + condition assembler

**Files:**
- Create: `starVLA/model/modules/geomem/condition_assembler.py` (the `geomem/__init__.py` already exists from Task 1)
- Test: `tests/geomemvla/test_condition_assembler.py`

**Interfaces:**
- Consumes: per-stream tensors of shape `[B, N_i, H_i]`.
- Produces:
  - `class ConditionAssembler(nn.Module)`
  - `__init__(self, stream_dims: dict[str, int], out_dim: int)` — builds one `nn.Linear(H_i, out_dim)` per named stream.
  - `forward(self, streams: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]` — returns `(cond [B, sum(N_i), out_dim], attention_mask [B, sum(N_i)] bool)`. Streams with value `None` are skipped (ablation).

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_condition_assembler.py
# [Geo-MemoryVLA] Unit test for the multi-stream condition assembler.
import torch
from starVLA.model.modules.geomem.condition_assembler import ConditionAssembler


def test_projects_and_concats_streams():
    asm = ConditionAssembler(stream_dims={"geo": 1024, "sem": 2048}, out_dim=256)
    streams = {"geo": torch.randn(2, 5, 1024), "sem": torch.randn(2, 3, 2048)}
    cond, mask = asm(streams)
    assert cond.shape == (2, 8, 256)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool and mask.all()


def test_skips_none_streams_for_ablation():
    asm = ConditionAssembler(stream_dims={"geo": 1024, "sem": 2048}, out_dim=256)
    streams = {"geo": torch.randn(2, 5, 1024), "sem": None}
    cond, mask = asm(streams)
    assert cond.shape == (2, 5, 256)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_condition_assembler.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/modules/geomem/condition_assembler.py
# [Geo-MemoryVLA] Projects each stream (geometry/semantic/memory/imagination)
# to a shared dim and concatenates into the [B, L, H] condition the GR00T head
# consumes. Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionAssembler(nn.Module):
    def __init__(self, stream_dims: dict[str, int], out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.ModuleDict({name: nn.Linear(d, out_dim) for name, d in stream_dims.items()})

    def forward(self, streams: dict[str, torch.Tensor]):
        parts, masks = [], []
        for name, x in streams.items():
            if x is None:
                continue
            p = self.proj[name](x)  # [B, N_i, out_dim]
            parts.append(p)
            masks.append(torch.ones(p.shape[0], p.shape[1], dtype=torch.bool, device=p.device))
        cond = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        return cond, mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_condition_assembler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/geomem/ tests/geomemvla/test_condition_assembler.py
git commit -m "[Geo-MemoryVLA] Add multi-stream condition assembler"
```

---

## Task 3: Dataloader episode_id + timestep (Phase B prerequisite)

**Files:**
- Modify: `starVLA/dataloader/gr00t_lerobot/datasets.py` (sample dict at ~`:1393-1416`)
- Test: `tests/geomemvla/test_dataloader_episode_fields.py`

**Interfaces:**
- Consumes: existing `trajectory_id, base_index` already computed at `datasets.py:1374` (`trajectory_id, base_index = self.all_steps[index]`).
- Produces: each sample dict additionally contains `"episode_id": int` and `"timestep": int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_dataloader_episode_fields.py
# [Geo-MemoryVLA] Guards that the LIBERO sample dict carries episode_id + timestep,
# required by the dual memory bank.
import ast
import pathlib


def test_sample_dict_includes_episode_and_timestep():
    src = pathlib.Path("starVLA/dataloader/gr00t_lerobot/datasets.py").read_text()
    # The sample dict literal must reference these keys.
    assert '"episode_id"' in src, "episode_id not added to sample dict"
    assert '"timestep"' in src, "timestep not added to sample dict"
    # And it must be parseable Python (no syntax break from the edit).
    ast.parse(src)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_dataloader_episode_fields.py -v`
Expected: FAIL — keys not present.

- [ ] **Step 3: Make the minimal edit**

In `starVLA/dataloader/gr00t_lerobot/datasets.py`, locate the sample dict (currently):

```python
        sample = {
            "action": action,
            "image": step_images,
            "lang": language,
            "robot_tag": self.tag,
        }
```

Replace with:

```python
        # [Geo-MemoryVLA] episode_id + timestep let the dual memory bank key
        # history per trajectory; trajectory_id/base_index already computed above.
        sample = {
            "action": action,
            "image": step_images,
            "lang": language,
            "robot_tag": self.tag,
            "episode_id": int(trajectory_id),
            "timestep": int(base_index),
        }
```

(If the exact dict differs at edit time, add the two keys to whatever sample dict `__getitem__` returns, using the `trajectory_id, base_index` variables from `self.all_steps[index]`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_dataloader_episode_fields.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add starVLA/dataloader/gr00t_lerobot/datasets.py
git add -f tests/geomemvla/test_dataloader_episode_fields.py
git commit -m "[Geo-MemoryVLA] Expose episode_id + timestep in LIBERO sample dict"
```

---

## Task 4: Dual memory bank (Phase B)

**Files:**
- Create: `starVLA/model/modules/memory/__init__.py` (empty)
- Create: `starVLA/model/modules/memory/memory_bank.py` (single-space bank, lifted)
- Create: `starVLA/model/modules/memory/dual_memory_bank.py` (geo + sem wrapper)
- Test: `tests/geomemvla/test_memory_bank.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone nn.Modules).
- Produces:
  - `class MemoryBank(nn.Module)`:
    - `__init__(self, token_dim: int, mem_length: int = 16, retrieval_layers: int = 2, fusion_type: str = "gate", consolidate_type: str = "tome", use_timestep_pe: bool = True)`
    - `process_batch(self, tokens: torch.Tensor, episode_ids: list[int], timesteps: list[int]) -> torch.Tensor` — `tokens [B, N, token_dim]` → fused `[B, N, token_dim]`.
    - `reset(self) -> None`
  - `class DualMemoryBank(nn.Module)`:
    - `__init__(self, geo_dim: int, sem_dim: int, **bank_kwargs)`
    - `process(self, geo: torch.Tensor, sem: torch.Tensor, episode_ids, timesteps) -> tuple[torch.Tensor, torch.Tensor]` — returns `(m_geo, m_sem)`. Either stream may be `None` (ablation) → returns `None` for it.
    - `reset(self) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_memory_bank.py
# [Geo-MemoryVLA] Unit tests for the lifted MemoryVLA memory bank (dim-agnostic)
# and the dual (geometry + semantic) wrapper.
import torch
from starVLA.model.modules.memory.memory_bank import MemoryBank
from starVLA.model.modules.memory.dual_memory_bank import DualMemoryBank


def test_single_bank_preserves_shape_and_builds_history():
    bank = MemoryBank(token_dim=16, mem_length=4)
    t0 = torch.randn(1, 3, 16)
    out0 = bank.process_batch(t0, episode_ids=[0], timesteps=[0])
    assert out0.shape == (1, 3, 16)
    # Second step in same episode retrieves prior memory (output stays well-formed).
    t1 = torch.randn(1, 3, 16)
    out1 = bank.process_batch(t1, episode_ids=[0], timesteps=[1])
    assert out1.shape == (1, 3, 16)
    assert torch.isfinite(out1).all()


def test_consolidation_caps_memory_length():
    bank = MemoryBank(token_dim=8, mem_length=3, consolidate_type="tome")
    for ts in range(10):
        bank.process_batch(torch.randn(1, 2, 8), episode_ids=[7], timesteps=[ts])
    assert len(bank.bank[7]) <= 3


def test_dual_bank_handles_none_stream():
    dual = DualMemoryBank(geo_dim=16, sem_dim=32, mem_length=4)
    geo = torch.randn(1, 5, 16)
    m_geo, m_sem = dual.process(geo, None, episode_ids=[0], timesteps=[0])
    assert m_geo.shape == (1, 5, 16)
    assert m_sem is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_memory_bank.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/modules/memory/memory_bank.py
# [Geo-MemoryVLA] Dimension-agnostic temporal memory bank.
# Adapted from MemoryVLA (MIT): vla/memory_vla.py:CogMemBank — retrieval via
# cross-attention, gate fusion, ToMe consolidation. Part of the Geo-MemoryVLA
# architecture — see docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        a, _ = self.attn(self.norm(q), kv, kv)
        return q + a


class _GateFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, cur, mem):
        scale = torch.sigmoid(self.gate(torch.cat([cur, mem], dim=-1)))
        return scale * cur + (1.0 - scale) * mem


class MemoryBank(nn.Module):
    def __init__(
        self,
        token_dim: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        use_timestep_pe: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.mem_length = mem_length
        self.consolidate_type = consolidate_type
        self.use_timestep_pe = use_timestep_pe
        self.retrieval = nn.ModuleList([_CrossBlock(token_dim) for _ in range(retrieval_layers)])
        self.fusion = _GateFusion(token_dim) if fusion_type == "gate" else None
        # episode_id -> list of (timestep:int, feat:[N, D])
        self.bank: dict[int, list[tuple[int, torch.Tensor]]] = {}

    def reset(self) -> None:
        self.bank = {}

    def _timestep_pe(self, ts: int, n: int, d: int, device) -> torch.Tensor:
        if not self.use_timestep_pe:
            return torch.zeros(n, d, device=device)
        pos = torch.full((d,), float(ts), device=device)
        i = torch.arange(d, device=device)
        pe = torch.where(i % 2 == 0, torch.sin(pos / 10000 ** (i / d)), torch.cos(pos / 10000 ** (i / d)))
        return pe.unsqueeze(0).expand(n, d)

    def _consolidate(self, eid: int) -> None:
        hist = self.bank[eid]
        while len(hist) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[eid] = hist[-self.mem_length :]
                return
            # ToMe: merge the most-similar consecutive pair.
            best_i, best_sim = 0, -1e9
            for i in range(len(hist) - 1):
                a = hist[i][1].mean(0)
                b = hist[i + 1][1].mean(0)
                sim = F.cosine_similarity(a, b, dim=0).item()
                if sim > best_sim:
                    best_sim, best_i = sim, i
            ts_a, fa = hist[best_i]
            ts_b, fb = hist[best_i + 1]
            merged = (ts_b, 0.5 * (fa + fb))
            hist = hist[:best_i] + [merged] + hist[best_i + 2 :]
            self.bank[eid] = hist

    def process_batch(self, tokens: torch.Tensor, episode_ids, timesteps) -> torch.Tensor:
        b, n, d = tokens.shape
        outs = []
        for i in range(b):
            eid = int(episode_ids[i])
            ts = int(timesteps[i])
            cur = tokens[i : i + 1]  # [1, N, D]
            hist = self.bank.get(eid, [])
            if hist:
                feats = []
                for h_ts, h_feat in hist:
                    pe = self._timestep_pe(h_ts, h_feat.shape[0], d, tokens.device)
                    feats.append(h_feat.to(tokens.device) + pe)
                mem = torch.cat(feats, dim=0).unsqueeze(0)  # [1, T*N, D]
                q = cur
                for blk in self.retrieval:
                    q = blk(q, mem)
                fused = self.fusion(cur, q) if self.fusion is not None else 0.5 * (cur + q)
            else:
                fused = cur
            outs.append(fused)
            # Write current observation into memory.
            self.bank.setdefault(eid, []).append((ts, tokens[i].detach().clone()))
            self._consolidate(eid)
        return torch.cat(outs, dim=0)
```

```python
# starVLA/model/modules/memory/dual_memory_bank.py
# [Geo-MemoryVLA] Dual (geometry + semantic) memory: VGGT latent in the geometric
# bank, Qwen3-VL cognitive token in the semantic bank — the perceptual/cognitive
# split from MemoryVLA, moved into a dual-latent space. Part of the Geo-MemoryVLA
# architecture — see docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch.nn as nn

from .memory_bank import MemoryBank


class DualMemoryBank(nn.Module):
    def __init__(self, geo_dim: int, sem_dim: int, **bank_kwargs):
        super().__init__()
        self.geo = MemoryBank(token_dim=geo_dim, **bank_kwargs)
        self.sem = MemoryBank(token_dim=sem_dim, **bank_kwargs)

    def reset(self) -> None:
        self.geo.reset()
        self.sem.reset()

    def process(self, geo, sem, episode_ids, timesteps):
        m_geo = self.geo.process_batch(geo, episode_ids, timesteps) if geo is not None else None
        m_sem = self.sem.process_batch(sem, episode_ids, timesteps) if sem is not None else None
        return m_geo, m_sem
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_memory_bank.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/memory/ tests/geomemvla/test_memory_bank.py
git commit -m "[Geo-MemoryVLA] Add dual (geometry+semantic) memory bank"
```

---

## Task 5: Imagination adapter over vendored VGGTWorldModel (Phase C)

**What this is:** NOT a hand-written imaginer. We wrap the vendored `VGGTWorldModel`
(`modules/vggt_world/model.py`, from Task 1) so Geo-MemoryVLA can (a) compute its imagination
loss during training and (b) produce an imagined future `GeometryState` (the 3D subgoal) at
inference. The flow transformer, z-prediction, SNR-weighted stage-1 loss, stage-2 flow-forcing,
and Euler rollout all live in the vendored code — we do not reimplement them.

**Files:**
- Create: `starVLA/model/modules/geomem/imagination_adapter.py`
- Test: `tests/geomemvla/test_imagination_adapter.py`

**Interfaces:**
- Consumes (vendored): `VGGTWorldModel(pretrained_vggt_repo=..., chunk_size, context_size, ...)`
  whose `forward(images[B,F,3,H,W], where=float) -> dict` returns `pred_state_tokens`,
  `target_state_tokens`, `stage`, `tau`; and `WorldModelLoss(...).forward(predictions, batch)
  -> {"objective": ...}`. For inference, `VGGTWorldModel.forward(images, ...)` in eval mode runs
  `_forecast` and returns `forecast_state_tokens` / decoded geometry.
- Produces:
  - `class ImaginationAdapter(nn.Module)`:
    - `__init__(self, pretrained_vggt_repo="facebook/VGGT-1B", chunk_size=2, context_size=2, latent_weight=1.0, decode_weights=(0.0,0.0,0.0))`
    - `training_loss(self, images: torch.Tensor, where: float = 0.0) -> torch.Tensor` — runs the
      world model forward + `WorldModelLoss`, returns the scalar `objective`.
    - `imagine_tokens(self, images: torch.Tensor, forecast_frames: int) -> torch.Tensor` — eval
      forecast; returns flattened imagined future tokens `[B, L_img, 1024]` (the subgoal condition).

> **Important wiring note for Task 6:** the vendored world model consumes **multi-frame image
> windows**, not a single VGGT latent. Phase C therefore needs the dataloader to provide a short
> future-frame window per sample (current `context_size` + future `chunk_size` frames). If the
> LIBERO dataloader yields only the current frame, the adapter falls back to the **single-frame
> degenerate mode** (context=target=current) for the smoke path, and the true multi-frame
> supervision is enabled once the dataloader emits frame windows — tracked as a Phase-C
> follow-up, analogous to the episode_id/timestep work in Task 3.

- [ ] **Step 1: Write the failing test (no model download — assert API shape only)**

```python
# tests/geomemvla/test_imagination_adapter.py
# [Geo-MemoryVLA] The adapter wraps the vendored VGGTWorldModel; here we only assert
# the adapter module imports and the vendored loss contract is intact (no VGGT download).
import torch


def test_world_model_loss_contract():
    # Vendored WorldModelLoss returns an 'objective' scalar from latent tokens.
    from starVLA.model.modules.vggt_world.losses import WorldModelLoss

    loss = WorldModelLoss(latent_weight=1.0)
    preds = {
        "pred_state_tokens": torch.randn(2, 10, 1024),
        "target_state_tokens": torch.randn(2, 10, 1024),
        "stage": "stage2",
    }
    out = loss(preds, batch={})
    assert "objective" in out and out["objective"].dim() == 0


def test_adapter_imports():
    from starVLA.model.modules.geomem.imagination_adapter import ImaginationAdapter
    assert hasattr(ImaginationAdapter, "training_loss")
    assert hasattr(ImaginationAdapter, "imagine_tokens")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_imagination_adapter.py -v`
Expected: FAIL — `imagination_adapter` not found (the vendored-loss test should already pass).

- [ ] **Step 3: Write the adapter**

```python
# starVLA/model/modules/geomem/imagination_adapter.py
# [Geo-MemoryVLA] Adapter over the vendored VGGT-World VGGTWorldModel. Produces the
# imagination training loss and the imagined future GeometryState (3D visual subgoal).
# All flow-matching / z-prediction / flow-forcing logic lives in modules/vggt_world.
# Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.vggt_world.losses import WorldModelLoss
from starVLA.model.modules.vggt_world.model import VGGTWorldModel


class ImaginationAdapter(nn.Module):
    def __init__(
        self,
        pretrained_vggt_repo: str = "facebook/VGGT-1B",
        chunk_size: int = 2,
        context_size: int = 2,
        latent_weight: float = 1.0,
        decode_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        self.world_model = VGGTWorldModel(
            pretrained_vggt_repo=pretrained_vggt_repo,
            chunk_size=chunk_size,
            context_size=context_size,
        )
        d_depth, d_point, d_cam = decode_weights
        self.criterion = WorldModelLoss(
            latent_weight=latent_weight,
            decode_depth_weight=d_depth,
            decode_point_weight=d_point,
            decode_camera_weight=d_cam,
        )

    def training_loss(self, images: torch.Tensor, where: float = 0.0) -> torch.Tensor:
        self.world_model.train()
        preds = self.world_model(images, where=where)
        return self.criterion(preds, batch={})["objective"]

    @torch.no_grad()
    def imagine_tokens(self, images: torch.Tensor, forecast_frames: int) -> torch.Tensor:
        self.world_model.eval()
        out = self.world_model(images, forecast_frames=forecast_frames)
        # pred_state_tokens: flattened future tokens [B, L_img, 1024].
        return out["pred_state_tokens"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_imagination_adapter.py -v`
Expected: PASS (both tests).

> The real `training_loss` / `imagine_tokens` paths (download VGGT-1B, multi-frame windows) are
> exercised in the Task 8 GPU smoke test, Phase C.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/modules/geomem/imagination_adapter.py tests/geomemvla/test_imagination_adapter.py
git commit -m "[Geo-MemoryVLA] Add imagination adapter over vendored VGGTWorldModel"
```

---

## Task 6: GeoMemoryVLA framework orchestrator

**Files:**
- Create: `starVLA/model/framework/VLM4A/GeoMemoryVLA.py`
- Test: `tests/geomemvla/test_geomemvla_framework.py`

**Interfaces:**
- Consumes: `WorldStateAdapter` (Task 1), `ConditionAssembler` (Task 2), `DualMemoryBank` (Task 4), `ImaginationAdapter` (Task 5), starVLA `get_vlm_model`, `get_action_model`, `merge_framework_config`, `FRAMEWORK_REGISTRY`.
- Produces:
  - `@FRAMEWORK_REGISTRY.register("GeoMemoryVLA")`
  - `class GeoMemoryVLA(baseframework)` with `forward(examples) -> {"action_loss": ...(+ "imagination_loss")}` and `predict_action(examples) -> {"normalized_actions": np.ndarray}`.
  - `@dataclass GeoMemoryVLADefaultConfig` with `world_state`, `memory`, `imagination` sub-dicts + `enabled` flags.

- [ ] **Step 1: Write the failing test (config + registration, no model download)**

```python
# tests/geomemvla/test_geomemvla_framework.py
# [Geo-MemoryVLA] Verifies the framework registers and its default config carries
# the ablation switches. Heavy model construction is covered by the slow smoke test.
def test_framework_is_registered():
    import starVLA.model.framework.base_framework as bf
    bf._auto_import_framework_modules()
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    assert "GeoMemoryVLA" in FRAMEWORK_REGISTRY._registry


def test_default_config_has_ablation_switches():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import GeoMemoryVLADefaultConfig
    cfg = GeoMemoryVLADefaultConfig()
    assert "enabled" in cfg.memory
    assert "enabled" in cfg.imagination
    assert cfg.world_state["stream"] in ("geo_only", "sem_only", "dual")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_geomemvla_framework.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/framework/VLM4A/GeoMemoryVLA.py
# [Geo-MemoryVLA] Framework orchestrator: frozen VGGT world-state + Qwen3-VL
# semantic stream + dual memory + 3D imagination, fused into the [B,L,H] condition
# consumed by the unchanged GR00T flow-matching head. Skeleton from QwenGR00T.
# Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.modules.geomem.condition_assembler import ConditionAssembler
from starVLA.model.modules.geomem.imagination_adapter import ImaginationAdapter
from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
from starVLA.model.modules.memory.dual_memory_bank import DualMemoryBank
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class GeoMemoryVLADefaultConfig:
    name: str = "GeoMemoryVLA"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
        "vl_hidden_dim": 2048,
    })
    world_state: dict = field(default_factory=lambda: {
        "enabled": True, "stream": "dual",
        "model_name": "facebook/VGGT-1B", "layer_index": 4, "num_cameras": 2,
    })
    memory: dict = field(default_factory=lambda: {
        "enabled": True, "mem_length": 16, "retrieval_layers": 2,
        "fusion_type": "gate", "consolidate_type": "tome",
    })
    imagination: dict = field(default_factory=lambda: {
        "enabled": True, "horizon": 4, "depth": 4, "steps": 4, "loss_scale": 0.5,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B", "action_hidden_dim": 1024, "hidden_size": 1024,
        "add_pos_embed": True, "max_seq_len": 2048, "action_dim": 7, "state_dim": 7,
        "action_horizon": 8, "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
        "num_timestep_buckets": 1000, "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            "cross_attention_dim": 1024, "dropout": 0.2, "final_dropout": True,
            "interleave_self_attention": True, "norm_type": "ada_norm",
            "num_layers": 16, "output_dim": 1024, "positional_embeddings": None,
        },
    })


@FRAMEWORK_REGISTRY.register("GeoMemoryVLA")
class GeoMemoryVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(GeoMemoryVLADefaultConfig, config)
        fw = self.config.framework

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        sem_dim = int(self.qwen_vl_interface.model.config.hidden_size)

        self.use_geo = fw.world_state["enabled"] and fw.world_state["stream"] in ("geo_only", "dual")
        self.use_sem = fw.world_state["stream"] in ("sem_only", "dual")
        if self.use_geo:
            # Vendored frozen VGGT-World backbone (Task 1 adapter).
            self.world_state = WorldStateAdapter(pretrained_vggt_repo=fw.world_state["model_name"])
            geo_dim = self.world_state.hidden_size  # 1024
        else:
            geo_dim = sem_dim

        self.use_memory = fw.memory["enabled"]
        if self.use_memory:
            self.memory = DualMemoryBank(
                geo_dim=geo_dim, sem_dim=sem_dim,
                mem_length=fw.memory["mem_length"],
                retrieval_layers=fw.memory["retrieval_layers"],
                fusion_type=fw.memory["fusion_type"],
                consolidate_type=fw.memory["consolidate_type"],
            )

        self.use_imag = fw.imagination["enabled"] and self.use_geo
        if self.use_imag:
            # Vendored VGGTWorldModel wrapper (Task 5 adapter). chunk/context drive the
            # imagined horizon; it consumes multi-frame image windows (see Task 5 note).
            self.imaginer = ImaginationAdapter(
                pretrained_vggt_repo=fw.world_state["model_name"],
                chunk_size=int(fw.imagination["horizon"]),
                context_size=int(fw.imagination.get("context_size", 2)),
            )
        self.imag_loss_scale = float(fw.imagination["loss_scale"])
        self.imag_horizon = int(fw.imagination["horizon"])

        # Condition hidden = GR00T cross_attention_dim. Build assembler over active streams.
        cond_dim = int(fw.action_model.diffusion_model_cfg.cross_attention_dim)
        stream_dims = {}
        if self.use_geo:
            stream_dims["geo"] = geo_dim
            if self.use_memory:
                stream_dims["m_geo"] = geo_dim
            if self.use_imag:
                stream_dims["imag"] = geo_dim
        if self.use_sem:
            stream_dims["sem"] = sem_dim
            if self.use_memory:
                stream_dims["m_sem"] = sem_dim
        self.assembler = ConditionAssembler(stream_dims=stream_dims, out_dim=cond_dim)

        self.action_model = get_action_model(config=self.config)
        self.action_horizon = int(fw.action_model.action_horizon)

    # --- helpers -------------------------------------------------------------
    def _encode(self, examples):
        images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
        sem = out.hidden_states[-1] if self.use_sem else None  # [B, L, H]
        geo = None
        if self.use_geo:
            pix = self._build_vggt_pixels(images, device=out.hidden_states[-1].device)
            state = self.world_state.encode(pix)            # GeometryState (frozen)
            geo = self.world_state.flatten(state)            # [B, F*2*tokens, 1024]
        return geo, sem

    def _build_vggt_pixels(self, images, device):
        # images: List[B][List[PIL]] -> [B, num_views, 3, H, W] in [0,1].
        import torchvision.transforms.functional as TF
        batch = []
        for views in images:
            vs = [TF.to_tensor(v.convert("RGB")) for v in views]
            batch.append(torch.stack(vs, dim=0))
        return torch.stack(batch, dim=0).to(device)

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

    def _assemble(self, geo, sem, m_geo, m_sem, imag):
        streams = {}
        if self.use_geo:
            streams["geo"] = geo
            if self.use_memory:
                streams["m_geo"] = m_geo
            if self.use_imag:
                streams["imag"] = imag
        if self.use_sem:
            streams["sem"] = sem
            if self.use_memory:
                streams["m_sem"] = m_sem
        return self.assembler(streams)

    # --- training ------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs):
        geo, sem = self._encode(examples)
        episode_ids = [e.get("episode_id", 0) for e in examples]
        timesteps = [e.get("timestep", 0) for e in examples]

        m_geo = m_sem = None
        if self.use_memory:
            m_geo, m_sem = self.memory.process(geo, sem, episode_ids, timesteps)

        device = (sem if sem is not None else geo).device
        imag = None
        imag_loss = torch.zeros((), device=device)
        if self.use_imag:
            # The vendored world model consumes a multi-frame image window. When the
            # dataloader provides one ("image_window" key: context+chunk frames), pass it;
            # otherwise fall back to the current views (degenerate single-step, Task 5 note).
            # `where` drives the stage-1/stage-2 flow-forcing curriculum.
            window = self._build_image_window(examples, device=device)
            imag = self.imaginer.imagine_tokens(window, forecast_frames=self.imag_horizon)
            imag_loss = self.imaginer.training_loss(window, where=float(kwargs.get("where", 0.0)))

        cond, mask = self._assemble(geo, sem, m_geo, m_sem, imag)

        actions = [e["action"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None
        rep = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=cond.device, dtype=cond.dtype)
            actions_target = actions[:, -self.action_horizon :, :].repeat(rep, 1, 1)
            cond_r = cond.repeat(rep, 1, 1)
            mask_r = mask.repeat(rep, 1)
            state_r = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=cond.device, dtype=cond.dtype)
                state_r = state_t.repeat(rep, 1, 1)
            action_loss = self.action_model(cond_r, actions_target, state_r, encoder_attention_mask=mask_r)

        out = {"action_loss": action_loss}
        if self.use_imag:
            out["imagination_loss"] = self.imag_loss_scale * imag_loss
        return out

    # --- inference -----------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if type(examples) is not list:
            examples = [examples]
        for e in examples:
            e["image"] = to_pil_preserve(e["image"])
        train_obs = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs:
            for e in examples:
                e["image"] = resize_images(e["image"], target_size=train_obs)

        geo, sem = self._encode(examples)
        episode_ids = [e.get("episode_id", 0) for e in examples]
        timesteps = [e.get("timestep", 0) for e in examples]
        m_geo = m_sem = None
        if self.use_memory:
            m_geo, m_sem = self.memory.process(geo, sem, episode_ids, timesteps)
        imag = None
        if self.use_imag:
            window = self._build_image_window(examples, device=geo.device)
            imag = self.imaginer.imagine_tokens(window, forecast_frames=self.imag_horizon)
        cond, mask = self._assemble(geo, sem, m_geo, m_sem, imag)

        state = [e["state"] for e in examples] if "state" in examples[0] else None
        state_t = torch.from_numpy(np.array(state)).to(cond.device, dtype=cond.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(cond, state_t, encoder_attention_mask=mask)
        return {"normalized_actions": pred.detach().cpu().numpy()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_geomemvla_framework.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f starVLA/model/framework/VLM4A/GeoMemoryVLA.py tests/geomemvla/test_geomemvla_framework.py
git commit -m "[Geo-MemoryVLA] Add framework orchestrator with ablation switches"
```

---

## Task 7: Training config

**Files:**
- Create: `examples/LIBERO/train_files/geo_memoryvla_libero.yaml`
- Test: `tests/geomemvla/test_config_loads.py`

**Interfaces:**
- Consumes: the framework name `GeoMemoryVLA` and its default config keys.
- Produces: a YAML that `OmegaConf.load` parses and `merge_framework_config` accepts.

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_config_loads.py
# [Geo-MemoryVLA] The training config parses and selects the right framework.
from omegaconf import OmegaConf


def test_config_parses_and_names_framework():
    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    assert cfg.framework.name == "GeoMemoryVLA"
    assert cfg.framework.world_state.stream in ("geo_only", "sem_only", "dual")
    assert cfg.framework.action_model.action_dim == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_config_loads.py -v`
Expected: FAIL — file not found.

- [ ] **Step 3: Write the config** (copied from `starvla_cotrain_libero.yaml`, framework section swapped)

```yaml
# [Geo-MemoryVLA] LIBERO training config. Ablation via framework.{memory,imagination}.enabled
# and framework.world_state.stream. See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
run_id: geo_memoryvla
run_root_dir: playground/Checkpoints
seed: 42
wandb_entity: your_wandb_entity
wandb_project: llavavla
is_debug: false
version_id: "0.21"

framework:
  name: GeoMemoryVLA
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
  world_state:
    enabled: true
    stream: dual          # geo_only | sem_only | dual
    model_name: facebook/VGGT-1B
    layer_index: 4
    num_cameras: 2
  memory:
    enabled: true
    mem_length: 16
    retrieval_layers: 2
    fusion_type: gate
    consolidate_type: tome
  imagination:
    enabled: true
    horizon: 4
    depth: 4
    steps: 4
    loss_scale: 0.5
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 8

datasets:
  vlm_data:
    dataset_py: vlm_datasets
    dataformat: llava_json
    dataset_use: sharegpt4v_coco
    eval_dataset: sharegpt4v_coco
    data_flatten: false
    base_interval: 2
    max_pixels: 307200
    min_pixels: 784
    model_max_length: 2048
    model_type: qwen2.5vl
    per_device_batch_size: 4
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA
    data_mix: libero_all
    action_type: delta_qpos
    sequential_step_sampling: True   # [Geo-MemoryVLA] needed so memory sees ordered episodes
    per_device_batch_size: 8
    load_all_data_for_training: true
    video_backend: torchvision_av

trainer:
  max_train_steps: 100000
  num_warmup_steps: 5000
  save_interval: 5000
  eval_interval: 100
  learning_rate:
    base: 2.5e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: 'qwen_vl_interface'
  loss_scale:
    vla: 1.0
    vlm: 0.1
  max_grad_norm: 1.0
  weight_decay: 0.0
  logging_frequency: 10
  gradient_clipping: 1.0
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/test_config_loads.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f examples/LIBERO/train_files/geo_memoryvla_libero.yaml tests/geomemvla/test_config_loads.py
git commit -m "[Geo-MemoryVLA] Add LIBERO training config with ablation switches"
```

---

## Task 8: GPU smoke test (end-to-end construction + forward)

**Files:**
- Test: `tests/geomemvla/test_smoke_gpu.py`

**Interfaces:**
- Consumes: the full framework + config. Marked `@pytest.mark.slow` (downloads VGGT + Qwen3-VL, needs a GPU).

- [ ] **Step 1: Write the slow smoke test**

```python
# tests/geomemvla/test_smoke_gpu.py
# [Geo-MemoryVLA] End-to-end construction + one forward/predict on real weights.
# Run explicitly on GPUs 4-7:  CUDA_VISIBLE_DEVICES=4 python -m pytest ... -m slow
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.slow
def test_forward_and_predict_phase_a():
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework

    cfg = OmegaConf.load("examples/LIBERO/train_files/geo_memoryvla_libero.yaml")
    # Phase A: world-state only, isolate the new VGGT plumbing from memory/imagination.
    cfg.framework.memory.enabled = False
    cfg.framework.imagination.enabled = False
    model = build_framework(cfg).cuda()

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, (16, 7)).astype(np.float16),
        "state": np.random.uniform(-1, 1, (1, 7)).astype(np.float16),
        "image": [img, img],          # 2 views -> num_cameras=2
        "lang": "pick up the cup",
        "episode_id": 0, "timestep": 0,
    }
    out = model([sample, dict(sample, lang="open the drawer")])
    assert torch.isfinite(out["action_loss"])

    pred = model.predict_action([sample])
    assert pred["normalized_actions"].shape == (1, 8, 7)
```

- [ ] **Step 2: Run it (expected to validate the real pipeline)**

Run: `cd /workspace/tingting/starVLA && CUDA_VISIBLE_DEVICES=4 python -m pytest tests/geomemvla/test_smoke_gpu.py -v -m slow -s`
Expected: PASS. If the vendored `FrozenVGGTBackbone.encode_states` fails on the installed VGGT aggregator (internal API drift), reconcile against `/workspace/tingting/vggt-world/vggt/world_model/backbone.py` and the installed `vggt` source, then re-run.

- [ ] **Step 3: Run Phase B+C smoke (memory + imagination enabled)**

Edit the test (or add a sibling) enabling `memory`/`imagination`, run again on `CUDA_VISIBLE_DEVICES=5`. Confirm `out["imagination_loss"]` is present and finite.

- [ ] **Step 4: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f tests/geomemvla/test_smoke_gpu.py
git commit -m "[Geo-MemoryVLA] Add GPU end-to-end smoke test"
```

---

## Task 9: Pytest marker registration + docs pointer

**Files:**
- Modify or Create: `pytest.ini` (register the `slow` marker) — check if one exists first.
- Modify: `docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md` (add "Implemented — see plan" note).

- [ ] **Step 1: Register the slow marker**

If `pytest.ini` / `pyproject.toml [tool.pytest.ini_options]` exists, add under markers; else create `tests/geomemvla/pytest.ini`:

```ini
[pytest]
markers =
    slow: requires GPU and large model downloads (Geo-MemoryVLA smoke tests)
```

- [ ] **Step 2: Run the fast suite to confirm green**

Run: `cd /workspace/tingting/starVLA && python -m pytest tests/geomemvla/ -v -m "not slow"`
Expected: all non-slow tests PASS, no marker warnings.

- [ ] **Step 3: Commit**

```bash
cd /workspace/tingting/starVLA
git add -f tests/geomemvla/pytest.ini
git commit -m "[Geo-MemoryVLA] Register slow pytest marker"
```

---

## Self-Review notes

- **Spec coverage:** §0 marker → Global Constraints + every file header. §3 data flow → Task 6 `_encode`/`_assemble`. §4.1 VGGT world-state (vendored) → Task 1. §4.2 dual memory → Task 4. §4.3 imagination (vendored VGGTWorldModel) → Task 5. §4.4 Qwen3-VL reuse → Task 6 `_encode`. §4.5 GR00T reuse → Task 6 (head untouched). §4.6 orchestrator → Task 6. §4.7 config → Task 7. §5 ablation switches → Task 6 config + Task 7 yaml + Task 8 phase toggles. §6 phasing → Task 8 phase A/B+C. §7.1 episode/timestep blocker → Task 3. §9 provenance/license → Task 1 PROVENANCE.md + vendored headers. Covered.
- **Correction applied:** the imaginer is no longer hand-written. Tasks 1 & 5 vendor the real VGGT-World code (`modules/vggt_world/`) and add thin adapters; the previous `GeoFlowImaginer`/`VGGTWorldStateEncoder` toys are removed. This was triggered by the user challenging from-abstract fidelity.
- **Known follow-up (called out inline, not hidden):** the vendored `VGGTWorldModel` consumes multi-frame image **windows**. The LIBERO dataloader currently yields the current frame's views; until it emits a context+chunk window (an "image_window" key), `_build_image_window` degenerates to single-frame and imagination supervision is weak. Emitting real windows is the Phase-C dataloader follow-up (sibling to Task 3's episode/timestep work).
- **Type consistency:** `WorldStateAdapter.encode(images)->GeometryState` + `flatten(state)->[B,L_g,1024]` (Task 1) match Task 6 `_encode`; `ImaginationAdapter.training_loss(images, where)->scalar` / `imagine_tokens(images, forecast_frames)->[B,L_img,1024]` (Task 5) match Task 6; `process(geo, sem, episode_ids, timesteps)->(m_geo,m_sem)` (Task 4) matches Task 6; `forward(vl_embs, actions, state, encoder_attention_mask)` matches the verified GR00T signature; `ConditionAssembler.forward(streams)->(cond,mask)` (Task 2) matches `_assemble`.
