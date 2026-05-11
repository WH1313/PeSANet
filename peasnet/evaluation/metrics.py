"""Metrics for PeSANet sequence predictions."""

from __future__ import annotations

import torch


def _flatten_time(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {tuple(x.shape)}.")
    return x.reshape(x.shape[0], -1)


def rmse_per_timestep(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = _flatten_time(prediction - target)
    return torch.sqrt(torch.mean(diff * diff, dim=1))


def mae_per_timestep(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = _flatten_time(prediction - target)
    return torch.mean(torch.abs(diff), dim=1)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_flat = _flatten_time(prediction)
    target_flat = _flatten_time(target)
    numerator = torch.linalg.norm(pred_flat - target_flat, dim=1)
    denominator = torch.linalg.norm(target_flat, dim=1).clamp_min(eps)
    return numerator / denominator


def correlation_per_timestep(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_flat = _flatten_time(prediction)
    target_flat = _flatten_time(target)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_centered * target_centered, dim=1)
    denominator = torch.linalg.norm(pred_centered, dim=1) * torch.linalg.norm(target_centered, dim=1)
    return numerator / denominator.clamp_min(eps)
