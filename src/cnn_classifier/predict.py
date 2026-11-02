"""Classify your own images with a trained checkpoint.

    python -m cnn_classifier.predict path/to/cat.jpg path/to/ship.png

Images are resized to 32x32, so expect best results on photos where the
object fills most of the frame, like the CIFAR-10 training images.
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from cnn_classifier.data import CLASSES, eval_transform
from cnn_classifier.model import load_checkpoint
from cnn_classifier.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    preprocess = transforms.Compose([transforms.Resize((32, 32)), eval_transform()])

    for path in args.images:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(image).softmax(dim=1).squeeze(0)
        top = probs.topk(args.top)
        guesses = ", ".join(f"{CLASSES[i]} {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{path}: {guesses}")


if __name__ == "__main__":
    main()
