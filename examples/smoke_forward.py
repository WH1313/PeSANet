"""Run a lightweight PeSANet forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peasnet.models import PesaNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = PesaNet(
        step=args.steps,
        effective_step=list(range(args.steps)),
        modes_x=args.modes,
        modes_y=args.modes,
    ).to(device)
    init_state = torch.randn(args.batch_size, 2, args.grid_size, args.grid_size, device=device)

    with torch.no_grad():
        sequence, second_last = model(init_state)

    print(f"sequence_shape={list(sequence.shape)}")
    print(f"second_last_shape={None if second_last is None else list(second_last.shape)}")
    print(f"num_parameters={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
