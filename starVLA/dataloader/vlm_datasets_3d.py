# [3DVLA-Stage1] Materialized-mixture VLM dataset with focal-length unification.
"""Stage-1 dataset for the 3D-capability-transfer project. Differences from
`vlm_datasets.py` (kept import-compatible; collator/rope reused from there):

1. Annotations come from ONE pre-shuffled materialized jsonl
   (`cfg.datasets.vlm_data.annotation_path`, built by materialize_variant_data.py).
   No per-rank reshuffle: every rank must see the identical order so accelerate
   batch sharding never duplicates samples across ranks.
2. Focal unification at load time (DATA_SPEC "全局不变量" #2/#3):
     - single-image regime (<=4 imgs): records with K are rescaled so fx ->
       F_TARGET=600; no-K records (general/embodied) pass through natively.
     - video regime (>4 imgs, generic3d): K records rescale to
       F_TARGET_VIDEO=280; no-K frames (scannetpp) rescale to the same
       per-frame area, keeping one consistent video sub-regime.
   After scaling we ASSERT the image fits the processor pixel budget so the
   Qwen smart_resize can never silently rescale again (the exact failure mode
   F=600 was chosen to avoid).
3. Zero-supervision guard: samples whose labels would be entirely -100 within
   model_max_length raise (the base __getitem__ retry then skips to i+1) —
   protects the loss from all-ignore NaN batches.
4. Audit counters (per worker process): slice/token/truncation/label-coverage
   stats printed every AUDIT_EVERY items and dumped by the smoke-audit tool.
"""
import json
import math
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.vlm_datasets import (
    IGNORE_INDEX,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    rank0_print,
    read_jsonl,
    update_processor_pixels,
)

F_TARGET = 600.0          # single-image regime (DATA_SPEC invariant #2)
F_TARGET_VIDEO = 280.0    # >4-image regime (generic3d 32-frame walkthroughs)
VIDEO_REGIME_MIN_IMGS = 5
VIDEO_FRAME_AREA = 72_000  # px target for no-K video frames (matches F=280 on scannet)
SINGLE_NOK_AREA = 378_000  # our own cap for no-K single images (coords resize-invariant)
SMART_RESIZE_MARGIN = 1.06  # Qwen rounds H,W up to multiples of 28: <=6% area growth
# Largest F=600-unified image: LIBERO wrist fx=312.77 @640x480 -> x1.918 ->
# 1228x921 -> smart-resized 1232x924 = 1,138,368 px. Config must grant it.
REQUIRED_MAX_PIXELS = 1_254_400  # = 1600 * 28 * 28
AUDIT_EVERY = 500


class Stage1MixtureDataset(LazySupervisedDataset):
    """Loads the materialized mixture and applies focal unification."""

    def __init__(self, processor, data_args):
        # Deliberately NOT calling super().__init__ (it wires the llava_json
        # registry + reshuffles); replicate only what we need.
        torch.utils.data.Dataset.__init__(self)
        from starVLA.dataloader.qwenvl_llavajson.rope2d import get_rope_index_3

        assert data_args.model_type == "qwen3vl", \
            f"Stage-1 targets Qwen3-VL; got model_type={data_args.model_type}"
        self.get_rope_index = get_rope_index_3
        self.model_type = data_args.model_type

        ann_path = data_args.annotation_path
        self.list_data_dict = read_jsonl(ann_path)
        rank0_print(f"[Stage1] {len(self.list_data_dict):,} conversations from {ann_path}")

        processor = update_processor_pixels(processor, data_args)
        # Contract: pixel budget must hold the largest unified single image
        # (663x497 -> smart-resized 672x504 = 338,688 px). See DATA_SPEC.
        ip = processor.image_processor
        self._max_pixels = getattr(ip, "max_pixels", None) or (
            ip.size.get("longest_edge") if isinstance(getattr(ip, "size", None), dict) else None)
        assert self._max_pixels and self._max_pixels >= REQUIRED_MAX_PIXELS, (
            f"max_pixels={self._max_pixels} would silently downscale F=600-unified "
            f"images (worst case LIBERO-wrist 1232x924); set "
            f"datasets.vlm_data.max_pixels>={REQUIRED_MAX_PIXELS}")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.item_fn = self._get_item

        self.audit = Counter()

    # ---- focal unification ------------------------------------------------
    def _load_unified_images(self, item):
        imgs = item["image"]
        if isinstance(imgs, str):
            imgs = [imgs]
        ks = item.get("Ks")
        if not ks:
            k = item.get("K")
            ks = [k] * len(imgs)
        assert len(ks) == len(imgs), f"Ks/image length mismatch: {len(ks)} vs {len(imgs)}"
        video_regime = len(imgs) >= VIDEO_REGIME_MIN_IMGS
        budget = self._max_pixels / SMART_RESIZE_MARGIN
        out = []
        for p, k in zip(imgs, ks):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            if k:
                s = (F_TARGET_VIDEO if video_regime else F_TARGET) / float(k[0])
                tag = "video_K" if video_regime else "single_K"
            elif video_regime:
                s = min(1.0, math.sqrt(VIDEO_FRAME_AREA / (w * h)))
                tag = "video_noK"
            else:
                # no metric supervision + resize-invariant coords: cap area
                # ourselves so raising max_pixels for the wrist case doesn't
                # inflate no-K token cost
                s = min(1.0, math.sqrt(SINGLE_NOK_AREA / (w * h)))
                tag = "single_noK"
            if abs(s - 1.0) > 1e-3:
                im = im.resize((max(28, round(w * s)), max(28, round(h * s))), Image.BICUBIC)
            area = im.size[0] * im.size[1]
            if area > budget:
                if k:  # F-unified metric record: a second resize would break F=600
                    raise AssertionError(
                        f"unified image {im.size} ({tag}, K={k}) exceeds pixel budget "
                        f"{budget:.0f}; raise max_pixels or fix the source record")
                # no-K record: no metric supervision, [0,2000)/[0,1] coords are
                # resize-invariant -> the processor's own downscale is harmless
                self.audit[f"img_{tag}_procresize"] += 1
            self.audit[f"img_{tag}"] += 1
            out.append(im)
        return out, video_regime

    # ---- tokenization with verified label masking --------------------------
    def _build_messages_unified(self, item):
        import re
        images, _ = self._load_unified_images(item)
        pool = list(images)
        messages = []
        for turn in item["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "user":
                content = []
                for seg in re.split(r"(<image>)", turn["value"]):
                    if seg == "<image>":
                        if not pool:
                            raise ValueError("<image> placeholders exceed images")
                        content.append({"type": "image", "image": pool.pop(0)})
                    elif seg.strip():
                        content.append({"type": "text", "text": seg.strip()})
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role,
                                 "content": [{"type": "text", "text": turn["value"]}]})
        if pool:
            raise ValueError(f"{len(pool)} image(s) not consumed by placeholders")
        return messages

    def _get_item(self, sources):
        assert len(sources) == 1
        item = sources[0]
        messages = self._build_messages_unified(item)
        res = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt")
        input_ids = res["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Unmask assistant answers: spans between '<|im_start|>assistant\n' and
        # '<|im_end|>'. Anchor on the full 2-token sequence (im_start, assistant)
        # instead of any lone 'assistant' token — a bare "assistant" inside user
        # text must NOT open a supervision span (bug in the base impl).
        ids = input_ids[0].tolist()
        im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = self.tokenizer.encode("assistant", add_special_tokens=False)
        assert len(assistant_id) == 1, "unexpected multi-token 'assistant'"
        assistant_id = assistant_id[0]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        L = len(ids)
        pos = 0
        n_spans = 0
        while pos < L - 2:
            if ids[pos] == im_start and ids[pos + 1] == assistant_id:
                ans_start = pos + 3  # im_start, 'assistant', '\n'
                ans_end = ans_start
                while ans_end < L and ids[ans_end] != im_end:
                    ans_end += 1
                stop = min(ans_end + 1, L)  # include <|im_end|>
                labels[0, ans_start:stop] = input_ids[0, ans_start:stop]
                n_spans += 1
                pos = ans_end
            pos += 1
        if n_spans == 0:
            raise ValueError("no assistant span found — chat template drift?")

        # zero-supervision guard vs collator truncation
        max_len = self.tokenizer.model_max_length
        if (labels[0, :max_len] != IGNORE_INDEX).sum().item() == 0:
            self.audit["skip_zero_supervision"] += 1
            raise ValueError(f"all labels beyond model_max_length={max_len} (seq {L})")
        if L > max_len:
            self.audit["truncated"] += 1

        res["labels"] = labels
        res["input_ids"] = input_ids

        grid = res.get("image_grid_thw")
        position_ids, _ = self.get_rope_index(
            self.merge_size, input_ids,
            image_grid_thw=grid if grid is not None else None)
        res["position_ids"] = position_ids
        res["attention_mask"] = [input_ids.size(1)]

        # audit trail
        sl = item.get("slice", "?")
        self.audit[f"convs_{sl}"] += 1
        self.audit[f"tokens_{sl}"] += L
        self.audit[f"sup_tokens_{sl}"] += int((labels != IGNORE_INDEX).sum())
        self.audit["items"] += 1
        if self.audit["items"] % AUDIT_EVERY == 0:
            print(f"[Stage1 audit pid={os.getpid()}] " +
                  " ".join(f"{k}={v}" for k, v in sorted(self.audit.items())))
        return res


def make_vlm_dataloader_3d(cfg):
    data_args = cfg.datasets.vlm_data
    processor = transformers.AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm)
    processor.tokenizer.model_max_length = int(data_args.model_max_length)
    processor.tokenizer.padding_side = "right"  # SFT with labels: pad right

    ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    dataset = Stage1MixtureDataset(processor, ns)
    collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=int(data_args.per_device_batch_size),
        collate_fn=collator,
        num_workers=int(getattr(data_args, "num_workers", 4)),
        pin_memory=True,
        shuffle=False,  # pre-shuffled materialized file; order must match across ranks
        drop_last=True,
    )
    return {"train_dataloader": loader}
