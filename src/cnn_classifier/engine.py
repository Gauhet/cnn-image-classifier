"""The training and evaluation loops, kept separate from the command-line scripts so tests can call them."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Run one pass over the training data. Returns (mean loss, accuracy).

    `scheduler` is stepped every batch (OneCycleLR expects that). Passing a
    `scaler` turns on mixed precision, which roughly halves training time on
    recent NVIDIA GPUs.
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Mean loss and accuracy on a dataset, with dropout and batch-norm updates switched off."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    for inputs, targets in tqdm(loader, desc="eval", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += loss_fn(outputs, targets).item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted and true labels for every example in `loader`, both on the CPU."""
    model.eval()
    preds, labels = [], []
    for inputs, targets in loader:
        preds.append(model(inputs.to(device)).argmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(preds), torch.cat(labels)
