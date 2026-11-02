"""CIFAR-10 loading: download, augmentation, normalisation and a fixed train/validation split."""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Per-channel mean and standard deviation of the CIFAR-10 training images.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# The Hugging Face mirror of CIFAR-10. It is much faster than the original
# University of Toronto host that torchvision downloads from.
SOURCE_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"


def load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images, labels) for "train" or "test" as uint8 (N, 32, 32, 3) and int64 (N,) arrays.

    The first call downloads the split and decodes every PNG once into a
    NumPy archive, so later runs load in about a second.
    """
    data_dir = Path(data_dir)
    cache = data_dir / f"cifar10_{split}.npz"
    if not cache.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        parquet = data_dir / f"cifar10_{split}.parquet"
        if not parquet.exists():
            print(f"downloading CIFAR-10 {split} split...")
            urllib.request.urlretrieve(SOURCE_URL.format(split=split), parquet)
        table = pq.read_table(parquet).to_pydict()
        images = np.stack([np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) for img in table["img"]])
        labels = np.array(table["label"], dtype=np.int64)
        np.savez(cache, images=images, labels=labels)
        parquet.unlink()

    with np.load(cache) as archive:
        return archive["images"], archive["labels"]


class CIFAR10(Dataset):
    """Wraps image and label arrays so each item comes out as (transformed image, label)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def train_transform() -> transforms.Compose:
    """Random crops and flips give the model slightly different images every epoch."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle 0..n-1 with a fixed seed and cut off the first `val_size` for validation."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[val_size:], order[:val_size]


def get_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation and test loaders, downloading CIFAR-10 on first use."""
    train_images, train_labels = load_split(data_dir, "train")
    test_images, test_labels = load_split(data_dir, "test")

    # Two views of the same 50k training images: augmented for training, plain for validation.
    train_full = CIFAR10(train_images, train_labels, train_transform())
    val_full = CIFAR10(train_images, train_labels, eval_transform())
    test_set = CIFAR10(test_images, test_labels, eval_transform())

    train_idx, val_idx = split_indices(len(train_full), val_size, seed)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(Subset(val_full, val_idx), shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader
