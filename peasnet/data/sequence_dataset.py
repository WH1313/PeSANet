"""Dataset wrapper for externally prepared sequence tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npy_sequences(path: str | Path) -> torch.Tensor:
    """Load a `.npy` sequence array without generating any data."""

    array = np.load(Path(path), mmap_mode=None).astype(np.float32)
    if array.ndim == 4:
        array = array[None, ...]
    if array.ndim != 5:
        raise ValueError(
            "Expected array shape [N, T, 2, H, W] or [T, 2, H, W], "
            f"got {array.shape}."
        )
    if array.shape[2] != 2:
        raise ValueError(f"Expected channel dimension size 2, got {array.shape[2]}.")
    return torch.from_numpy(array)


class SequenceDataset(Dataset):
    """Dataset for tensors shaped [num_samples, time, 2, height, width]."""

    def __init__(self, sequences: torch.Tensor) -> None:
        if sequences.ndim == 4:
            sequences = sequences.unsqueeze(0)
        if sequences.ndim != 5:
            raise ValueError(
                "Expected tensor shape [N, T, 2, H, W] or [T, 2, H, W], "
                f"got {tuple(sequences.shape)}."
            )
        if sequences.shape[2] != 2:
            raise ValueError(f"Expected channel dimension size 2, got {sequences.shape[2]}.")
        self.sequences = sequences.float()

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]
