import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_classifier.engine import evaluate, predict_all, train_one_epoch
from cnn_classifier.model import CifarCNN

CPU = torch.device("cpu")


def tiny_loader():
    torch.manual_seed(0)
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(images, labels), batch_size=16)


def test_model_can_memorise_a_tiny_dataset():
    # A sanity check that gradients flow: a working training loop should
    # drive the loss on 32 random images close to zero.
    torch.manual_seed(0)
    model = CifarCNN(widths=(16, 32), dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    loader = tiny_loader()

    first_loss, _ = train_one_epoch(model, loader, optimizer, loss_fn, CPU)
    for _ in range(30):
        train_one_epoch(model, loader, optimizer, loss_fn, CPU)
    final_loss, final_acc = evaluate(model, loader, loss_fn, CPU)

    assert final_loss < first_loss / 4
    assert final_acc > 0.9


def test_predict_all_returns_one_label_per_example():
    preds, labels = predict_all(CifarCNN(widths=(8,)), tiny_loader(), CPU)
    assert preds.shape == labels.shape == (32,)
