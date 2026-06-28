"""Lightweight ModalityConfig — no heavy dataloader imports.

Kept in its own module so data-config files (e.g. examples/LIBERO) can
import ModalityConfig without pulling in the full datasets.py import chain
(albumentations, pytorch3d, decord, …).
"""

from pydantic import BaseModel


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index.
    The returned data will correspond to the original data at a sampled
    base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
