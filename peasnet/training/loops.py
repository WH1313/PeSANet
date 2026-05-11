"""Minimal training utilities for externally supplied sequence data."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrainStats:
    loss: float
    batches: int


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> TrainStats:
    """Train one epoch using MSE over the model's saved sequence steps."""

    model.train()
    device = torch.device(device)
    total_loss = 0.0
    batches = 0

    for sequence in loader:
        sequence = sequence.to(device)
        optimizer.zero_grad(set_to_none=True)

        prediction, _ = model(sequence[:, 0])
        target = sequence[:, : prediction.shape[1]]
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1

    if batches == 0:
        raise ValueError("Training loader produced no batches.")
    return TrainStats(loss=total_loss / batches, batches=batches)
