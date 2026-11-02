from pathlib import Path

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two 3x3 convolutions with batch norm and ReLU, then 2x2 max pooling.

    Each block keeps the spatial size through the convolutions (padding=1) and
    halves it with pooling, so 32x32 inputs become 16x16, 8x8, then 4x4.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )


class CifarCNN(nn.Module):
    """A small VGG-style network: stacked conv blocks, global average pooling, one linear layer."""

    def __init__(
        self,
        num_classes: int = 10,
        widths: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {"num_classes": num_classes, "widths": list(widths), "dropout": dropout}

        blocks, in_channels = [], 3
        for width in widths:
            blocks.append(ConvBlock(in_channels, width, dropout))
            in_channels = width
        self.features = nn.Sequential(*blocks)

        # Averaging each feature map to one number keeps the classifier tiny
        # (256 -> 10) instead of flattening thousands of values into a big linear layer.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: CifarCNN, path: str | Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CifarCNN, dict]:
    """Rebuild the model from a checkpoint and return it with the checkpoint's metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = CifarCNN(config["num_classes"], tuple(config["widths"]), config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
