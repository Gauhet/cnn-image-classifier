"""Train the CNN on CIFAR-10.

    python -m cnn_classifier.train --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from cnn_classifier.data import get_loaders
from cnn_classifier.engine import evaluate, train_one_epoch
from cnn_classifier.model import CifarCNN, count_parameters, save_checkpoint
from cnn_classifier.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, help="peak learning rate for the one-cycle schedule")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--history", default="results/history.json")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on the GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, _ = get_loaders(args.data_dir, args.batch_size, num_workers=args.num_workers, seed=args.seed)

    model = CifarCNN(dropout=args.dropout).to(device)
    print(f"device: {device} | parameters: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay
    )
    # One-cycle: warm the learning rate up to --lr over the first 30% of steps,
    # then anneal it towards zero. Converges much faster than a fixed rate.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scheduler, scaler)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, args.checkpoint, epoch=epoch, val_acc=val_acc)
            marker = "  * saved"

        print(
            f"epoch {epoch:2d}/{args.epochs} | "
            f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.3f} acc {val_acc:.3f} | "
            f"{time.perf_counter() - start:.0f}s{marker}"
        )

    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).write_text(json.dumps(history, indent=2))
    print(f"best validation accuracy: {best_val_acc:.4f} -> {args.checkpoint}")


if __name__ == "__main__":
    main()
