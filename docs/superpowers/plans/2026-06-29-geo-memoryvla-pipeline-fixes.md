# Geo-MemoryVLA Pipeline Bug-Fix Plan

> Execution: INLINE by the controller (no subagents this round, per user). Fix order: M1 → S1 → B1 → B2/B3 → S2 → M2 → B4. Each fix is TDD with a test that exercises the REAL trainer / mixture-dataset / sampler path (the layer the per-task reviews + GPU smoke all bypassed — that blindness is why these survived).

**Goal:** Fix the 14 audited+verified bugs so Geo-MemoryVLA's two core contributions (3D imagination + temporal memory) actually train and run, not just produce finite smoke losses.

**Source of bugs (honest attribution):** of 6 Criticals, 4 live in pre-existing starVLA code or the new↔old seam (trainer drops the loss key; mixture dataloader omits episode_id; random sampler; no reset wiring), 2 in our GeoMemoryVLA.py (S1 view-as-frame, S2 inference crash). Root cause across the board = **test blindness**: nothing ran the real `LeRobotMixtureDataset` + real trainer. Every fix here adds a test at that layer.

**Env:** starVLA conda — `/workspace/ghsun/miniconda3/envs/starVLA/bin/python`. Run: `cd /workspace/tingting/starVLA && PYTHONPATH=/workspace/tingting/starVLA NO_ALBUMENTATIONS_UPDATE=1 <py> -m pytest ...`. GPU smoke on GPUs 0-3.

## Global Constraints
- `# [Geo-MemoryVLA]` marker on every changed block → this plan / the design specs.
- Tests must hit the REAL path, not hand-built dicts: `LeRobotMixtureDataset.__getitem__`, the trainer's loss-sum, `sample_step`. Use real LIBERO parquet under `playground/Datasets/LEROBOT_LIBERO_DATA` where the audit confirmed it exists.
- S1 decision (user-approved): **single-view (primary) time series first; multi-view is a later extension** (VGGT supports multi-view; defer it). So the world model gets a true temporal sequence, not view-interleaved frames.
- Don't touch the refuted non-bugs (no-op collate; gripper raw [0,1] — both verified intended).

---

## Fix M1 — sum imagination_loss into backward (Critical)
**File:** `starVLA/training/train_starvla.py:405-408` (and `train_starvla_cotrain.py:367-368, 395-397`).
**Bug:** `total_loss = action_loss` drops `imagination_loss`; flow transformer gets zero gradient (imagine_tokens is @no_grad) → never trains.
**Fix:** after reading action_loss, add any extra finite loss terms the model returns:
```python
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                # [Geo-MemoryVLA] include imagination_loss (and any future aux loss) so the
                # flow transformer actually receives gradient. Trainer previously dropped it.
                total_loss = action_loss
                imagination_loss = output_dict.get("imagination_loss", None)
                if imagination_loss is not None:
                    total_loss = total_loss + imagination_loss
```
Mirror in both cotrain branches (DeepSpeed `self.model.backward(total_loss)` + accelerate). Log `imagination_loss` in the metrics dict next to action_loss.
**Test:** `tests/geomemvla/test_pipeline_fixes.py::test_trainer_sums_imagination_loss` — construct the loss-sum logic in isolation (a tiny stand-in dict `{"action_loss": t1, "imagination_loss": t2}`) and assert the summed total requires grad through both; PLUS a source-grep assert that `imagination_loss` appears in train_starvla.py's step (guards regression). (Full trainer run is GPU; the source+logic test catches the drop.)

---

## Fix S1 — single-view temporal window (Critical, user-approved approach A)
**Files:** `examples/LIBERO/train_files/data_registry/data_config.py` (image_window modality_keys), `GeoMemoryVLA.py:_build_image_window`.
**Bug:** image_window uses BOTH video_keys → 5 frames × 2 views flattened into 10 "frames"; world model slices views-as-timesteps.
**Fix:**
1. data_config: the `image_window` ModalityConfig uses ONLY the primary camera key, not all video_keys:
```python
        # [Geo-MemoryVLA] image_window is a single-camera TEMPORAL sequence for the world
        # model (VGGT-World is a monocular-trajectory model). Multi-view imagination is a
        # deferred extension. Use primary only so frames == timesteps.
        image_window_keys = [self.video_keys[0]]  # "video.primary_image"
        ...
        configs["image_window"] = ModalityConfig(delta_indices=self.image_window_indices,
                                                 modality_keys=image_window_keys)
```
2. `_build_image_window`: now each frame has exactly 1 view → produce `[B, F, 3, H, W]` (F=5), not `[B, F*views, ...]`. Since the dataloader now yields `image_window` as `List[F][1]`, the existing loop already cats to `[B, F*1, 3, H, W]` = `[B, 5, 3, H, W]`. Add an assertion that the per-example frame count == `_imag_window_len` (turn the dead guard live):
```python
        out = torch.stack(windows, dim=0).to(device)
        # [Geo-MemoryVLA] world model frame axis must equal the temporal window length
        # (single-view => frames == timesteps). Turns the previously-dead _imag_window_len guard live.
        if getattr(self, "use_imag", False) and hasattr(self, "_imag_window_len"):
            assert out.shape[1] == self._imag_window_len, (
                f"image_window frame count {out.shape[1]} != expected {self._imag_window_len} "
                f"(single-view temporal window). See pipeline-fixes plan S1.")
        return out
```
**Test:** `test_single_view_window_is_temporal` — real `LeRobotMixtureDataset.__getitem__` (or get_step_data+_pack_sample) with enable_image_window → assert `sample["image_window"]` has F=5 frames each with exactly 1 view (primary), and that `sample["image"]` still has 2 views (observation untouched). PLUS `_build_image_window` on that sample yields `[B,5,3,H,W]`.

---

## Fix B1 — attach episode_id/timestep on the mixture path (Critical)
**File:** `datasets.py:2440-2444` (`LeRobotMixtureDataset.__getitem__`).
**Bug:** the real training path (mixture) never sets episode_id/timestep; only the unused single-dataset path does (our Phase-A Task 3 edited the wrong class). All samples default to (0,0) → memory collapses to one bucket.
**Fix:** at the return site, where `trajectory_id, step` are already in scope:
```python
                sample = dataset._pack_sample(data)
                # [Geo-MemoryVLA] attach episode/timestep on the REAL (mixture) path so the
                # dual memory bank keys history per trajectory. (Phase-A added these only on
                # the unused LeRobotSingleDataset path — see pipeline-fixes B1.)
                sample["episode_id"] = int(trajectory_id)
                sample["timestep"] = int(step)
                return sample
```
**Test:** `test_mixture_getitem_has_episode_timestep` — build a real mixture dataset on real parquet, call `__getitem__`, assert `sample["episode_id"]` and `sample["timestep"]` present and are ints (and vary across indices, not all 0).

---

## Fix B2/B3 — honor sequential_step_sampling for memory runs (Critical/Important)
**File:** `datasets.py:2375-2395` (`sample_step`), reads `self.data_cfg`.
**Bug:** `sample_step` RNG-picks (dataset, trajectory, step) independently; `sequential_step_sampling` flag is read by NO .py. Memory sees random, non-adjacent steps.
**Fix:** branch `sample_step` on the flag. When sequential, map the flat `index` to (trajectory, step) in episode order (the `all_steps`-style flattening already exists conceptually). Minimal correct version:
```python
    def sample_step(self, index):
        # [Geo-MemoryVLA] sequential mode: walk trajectories in episode order so the memory
        # bank sees temporally-ordered, contiguous steps. Default (random) preserved otherwise.
        seq = False
        if getattr(self, "data_cfg", None) is not None:
            seq = bool(self.data_cfg.get("sequential_step_sampling", False))
        if seq:
            return self._sequential_step(index)
        # ... existing random path unchanged ...
```
Add `_sequential_step(index)` that builds (once, cached) a flat list of (dataset_idx, trajectory_id, base_index) across all episodes in order, and indexes it by `index % total`. Also, **sort `process_batch` history by timestep** before PE/consolidation in `memory_bank.py` (so even mild out-of-order arrival is corrected):
```python
        # [Geo-MemoryVLA] order history by timestep before PE/ToMe (memory is temporal).
        hist = sorted(self.bank.get(eid, []), key=lambda x: x[0])
```
**Test:** `test_sequential_sampling_yields_ordered_steps` — with `data_cfg={"sequential_step_sampling": True}`, assert consecutive `sample_step(i)` for i in a trajectory's range yield increasing `base_index` within the same trajectory; and a `memory_bank` test asserting history is timestep-sorted after out-of-order inserts.

---

## Fix S2 — predict_action must not require image_window at inference (Critical)
**File:** `GeoMemoryVLA.py:281-283` → `_build_image_window:188-195`.
**Bug:** inference samples carry no image_window → `_build_image_window` raises → any imagination-trained ckpt crashes on first eval request.
**Fix:** at inference, build the window from the current frame replicated to the window length (a degenerate-but-valid temporal window), rather than raising. Keep the hard-raise for the TRAINING path (forward), where a missing window is a real misconfiguration. Concretely, give `_build_image_window` an `allow_degenerate` flag; `predict_action` passes True:
```python
    def _build_image_window(self, examples, device, allow_degenerate=False):
        ...
        for e in examples:
            if "image_window" not in e:
                if getattr(self, "use_imag", False) and not allow_degenerate:
                    raise ValueError(... training misconfig ...)
                # [Geo-MemoryVLA] inference: no rolling buffer → replicate current frame to
                # the window length so the world model gets a valid (static) temporal window.
                frames = [e["image"]] * getattr(self, "_imag_window_len", 5)
            else:
                frames = e["image_window"]
            ...
```
And `predict_action`: `window = self._build_image_window(examples, device=geo.device, allow_degenerate=True)`.
**Test:** `test_predict_action_no_window_does_not_raise` — `GeoMemoryVLA.__new__` bypass, `use_imag=True`, `_imag_window_len=5`, examples without image_window → `_build_image_window(..., allow_degenerate=True)` returns `[B,5,3,H,W]` and does NOT raise; and the training path (allow_degenerate=False) STILL raises.

---

## Fix M2 — thread training progress into `where` (Important)
**Files:** `GeoMemoryVLA.forward` (already reads `kwargs.get("where",0.0)`), `train_starvla.py:405` call site.
**Bug:** trainers call `forward(batch)` with no `where` → always 0.0 → Stage-2 flow-forcing never engages.
**Fix:** pass progress from the trainer:
```python
                where = self.completed_steps / max(1, self.config.trainer.max_train_steps)
                output_dict = self.model.forward(batch_vla, where=where)
```
(forward already plumbs `where` → imaginer.training_loss → world_model. No model change needed beyond confirming `forward(self, examples=None, **kwargs)` accepts it — it does.) Mirror in cotrain.
**Test:** `test_where_progresses` — assert forward signature accepts `where`; logic test that `completed_steps/max_train_steps` crosses 0.5 (stage2_start) partway through training (so Stage-2 engages). Source-grep that the trainer passes `where=`.

---

## Fix B4 — reset memory at episode/eval boundaries (Important)
**Files:** `GeoMemoryVLA.predict_action` (reset between eval rollouts), trainer epoch boundary (or document non-persistence).
**Bug:** `memory.reset()` never called → keys grow unbounded, stale memory bleeds across epochs and across eval rollouts.
**Fix:**
- `predict_action`: when a new episode starts (e.g. `examples[0].get("timestep", 0) == 0` or an explicit `reset` kwarg from the eval client), call `self.memory.reset()` before processing. Simplest robust trigger: reset when timestep==0.
```python
        # [Geo-MemoryVLA] new-episode boundary at inference: clear stale cross-rollout memory.
        if self.use_memory and examples[0].get("timestep", 0) == 0:
            self.memory.reset()
```
- Training: document that the bank self-bounds per episode_id via mem_length; add an optional `memory.reset()` at epoch start in the trainer loop (guarded by `hasattr(model, "memory")`). Keep minimal.
**Test:** `test_memory_reset_on_new_episode` — drive `predict_action`-like flow (or directly the reset trigger) with timestep==0 after populating the bank; assert bank is cleared. (Unit-level on the memory bank + the trigger condition.)

---

## Cleanup (fold in opportunistically, not separate tasks)
- Sync `image_window_context/chunk` from `framework.imagination` in the gate (Minor "window size never synced") — set `_dc.image_window_context/chunk` alongside `enable_image_window` in `dataloader/__init__.py`.
- Remove/repurpose the dead `_imag_window_len` comment now that S1 makes it a live assertion.
- `CachedLeRobotSingleDataset.get_step_data` missing `_apply_action_mode` — add the call (latent; only bites delta/rel action modes).

---

## Self-Review
- Coverage: every confirmed bug (M1,S1,B1,B2,B3,S2,M2,B4) has a fix + a real-path test. Refuted non-bugs untouched. Minors folded into cleanup.
- The recurring root cause (no real-trainer/mixture test) is addressed: each fix's test hits the real layer, not a hand-built dict or smoke.
- Order M1→S1→B1→B2/B3→S2→M2→B4 matches the audit's dependency note (imagination must train before its representation/inference matter; memory keying before ordering).
- After all fixes: re-run the full fast suite + a geo_only+imagination+memory GPU smoke that goes through a REAL mini mixture dataset (not hand-built), to prove the seam end-to-end.
