"""Shared CLI argument parser for training scripts."""

import argparse


def parse_train_args(description="Train a model"):
    """Return an ArgumentParser with common training flags.

    Scripts call this helper and optionally add script-specific flags.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience (0 to disable)")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cuda/cpu); auto-detect if omitted")
    parser.add_argument("--save-path", type=str, default=None, help="Checkpoint save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: run without saving checkpoints (sets patience=0). "
                             "Use for quick sanity checks so a 1-epoch run does not clobber a "
                             "real checkpoint at the same path.")
    return parser
