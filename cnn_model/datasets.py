"""This module downloads and manipulates datasets"""
from typing import Tuple

import torch
import torchvision

from cnn_model.common import DATACACHEDIR


def get_mnist(
    batch_size: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get MNIST dataset"""

    DATACACHEDIR.mkdir(parents=True, exist_ok=True)

    training_data = torchvision.datasets.MNIST(
        root=DATACACHEDIR,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root=DATACACHEDIR,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def extract_shape(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Size, int]:
    """Extract the tensor shape and number of labels in a dataloader"""

    shape_set = set(tensor.shape for tensor, _ in dataloader.dataset)
    if len(shape_set) < 1:
        raise RuntimeError("Empty Dataloader")
    if len(shape_set) > 1:
        raise RuntimeError("Inconsistent Dataloader")

    label_set = set(label for _, label in dataloader.dataset)

    return shape_set.pop(), len(label_set)


def get_input_parameters(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
) -> Tuple[int, int, int]:
    """Get the parameters of the input"""

    train_shape, train_label_count = extract_shape(train_dataloader)
    test_shape, test_label_count = extract_shape(test_dataloader)

    if train_shape != test_shape:
        raise RuntimeError("Mismatched test shape")

    if train_label_count != test_label_count:
        raise RuntimeError("Mismatched test label count")

    if len(train_shape) != 3:
        raise RuntimeError("Bad input shape")

    if train_shape[2] != train_shape[1]:
        raise RuntimeError("Non-square input")

    return train_shape[0], train_shape[1], train_label_count
