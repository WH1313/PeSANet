"""Train PeSANet on an externally prepared `.npy` sequence file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peasnet.data import SequenceDataset, load_npy_sequences
from peasnet.models import PesaNet
from peasnet.training import train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npy", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--modes-x", type=int, default=8)
    parser.add_argument("--modes-y", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sequences = load_npy_sequences(args.train_npy)
    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PesaNet(
        step=args.steps,
        effective_step=list(range(args.steps)),
        modes_x=args.modes_x,
        modes_y=args.modes_y,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        stats = train_one_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch + 1} loss={stats.loss:.8f} batches={stats.batches}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )


if __name__ == "__main__":
    main()
