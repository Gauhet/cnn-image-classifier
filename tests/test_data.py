import numpy as np
from PIL import Image

from cnn_classifier.data import CIFAR10, eval_transform, split_indices, train_transform


def test_dataset_returns_transformed_image_and_int_label():
    images = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(5)
    dataset = CIFAR10(images, labels, eval_transform())
    image, label = dataset[3]
    assert len(dataset) == 5
    assert image.shape == (3, 32, 32)
    assert label == 3 and isinstance(label, int)


def test_transforms_produce_normalised_tensors():
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    for transform in (train_transform(), eval_transform()):
        tensor = transform(image)
        assert tensor.shape == (3, 32, 32)
        # The red channel is far above the dataset mean, so it should end up well above zero.
        assert tensor[0].mean() > 1


def test_split_is_disjoint_complete_and_repeatable():
    train, val = split_indices(100, 20, seed=1)
    assert len(val) == 20 and len(train) == 80
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(100))
    assert split_indices(100, 20, seed=1) == (train, val)
