import torch

from cnn_classifier.model import CifarCNN, count_parameters, load_checkpoint, save_checkpoint


def test_output_has_one_score_per_class():
    model = CifarCNN()
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_parameter_count_stays_small():
    # Roughly 1.1M parameters: big enough for ~90% on CIFAR-10, small enough to train in minutes.
    assert 500_000 < count_parameters(CifarCNN()) < 2_000_000


def test_checkpoint_round_trip(tmp_path):
    model = CifarCNN(widths=(8, 16), dropout=0.1).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, epoch=3, val_acc=0.5)

    loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(model(x), loaded(x))
    assert checkpoint["epoch"] == 3
