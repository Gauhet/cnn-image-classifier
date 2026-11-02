"""Evaluate the best checkpoint on the CIFAR-10 test set and draw the README figures.

    python -m cnn_classifier.evaluate
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from cnn_classifier.data import CLASSES, MEAN, STD, get_loaders
from cnn_classifier.engine import predict_all
from cnn_classifier.model import load_checkpoint
from cnn_classifier.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--history", default="results/history.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def plot_history(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set(title="Loss", xlabel="epoch")
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set(title="Accuracy", xlabel="epoch")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(dataset, preds: torch.Tensor, path: Path, count: int = 16) -> None:
    """A grid of test images titled with the prediction; wrong ones show the true label in red."""
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, count // 2, figsize=(count * 0.9, 4))
    for i, ax in enumerate(axes.flat):
        image, label = dataset[i]
        image = (image * std + mean).clamp(0, 1).permute(1, 2, 0)
        pred = preds[i].item()
        ax.imshow(image)
        ax.set_title(
            CLASSES[pred] if pred == label else f"{CLASSES[pred]}\n(true: {CLASSES[label]})",
            fontsize=8, color="black" if pred == label else "red",
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    assets = Path(args.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(args.checkpoint, device)
    _, _, test_loader = get_loaders(args.data_dir, num_workers=args.num_workers)

    preds, labels = predict_all(model, test_loader, device)
    accuracy = (preds == labels).float().mean().item()
    per_class = {
        name: (preds[labels == i] == i).float().mean().item() for i, name in enumerate(CLASSES)
    }

    print(f"test accuracy: {accuracy:.4f} (checkpoint from epoch {checkpoint['epoch']})")
    for name, acc in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"  {name:<10} {acc:.3f}")

    metrics = {"test_accuracy": accuracy, "best_epoch": checkpoint["epoch"], "per_class_accuracy": per_class}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(labels, preds, assets / "confusion_matrix.png")
    plot_samples(test_loader.dataset, preds, assets / "sample_predictions.png")
    if Path(args.history).exists():
        plot_history(json.loads(Path(args.history).read_text()), assets / "training_curves.png")
    print(f"figures written to {assets}/")


if __name__ == "__main__":
    main()
