"""This module downloads and manipulates datasets"""
from pathlib import Path
from typing import Dict
from typing import Tuple

import torch
import torchvision  # type:ignore[import-untyped]

_CACHED_PARAMS: Dict[str, Tuple[int, int, int]] = {}
DATACACHEDIR = Path("cache/data")


def get_dataset(
    name: str = "MNIST",
    batch_size: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get a dataset by name"""

    DATACACHEDIR.mkdir(parents=True, exist_ok=True)

    train_args = {
        "root": DATACACHEDIR,
        "train": True,
        "download": True,
        "transform": torchvision.transforms.ToTensor(),
    }

    test_args = {
        "root": DATACACHEDIR,
        "train": False,
        "download": True,
        "transform": torchvision.transforms.ToTensor(),
    }

    if name == "MNIST":
        train_data = torchvision.datasets.MNIST(**train_args)
        test_data = torchvision.datasets.MNIST(**test_args)
    elif name == "CIFAR10":
        train_data = torchvision.datasets.CIFAR10(**train_args)
        test_data = torchvision.datasets.CIFAR10(**test_args)
    else:
        raise ValueError(f"Unknown dataset {name}")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def extract_shape(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Size, int]:
    """Extract the tensor shape and number of labels in a dataloader"""

    shape_set = set(tensor.shape for tensor, _ in dataloader.dataset)  # type:ignore
    if len(shape_set) < 1:
        raise RuntimeError("Empty Dataloader")
    if len(shape_set) > 1:
        raise RuntimeError("Inconsistent Dataloader")

    label_set = set(label for _, label in dataloader.dataset)  # type:ignore

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


def get_dataset_and_params(
    name: str = "MNIST",
    batch_size: int = 1,
) -> Tuple[
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    Tuple[int, int, int],
]:
    """Get a dataset and its (cached) parameters by name"""

    dataset = get_dataset(name=name, batch_size=batch_size)

    if name in _CACHED_PARAMS:
        params = _CACHED_PARAMS[name]
    else:
        params = get_input_parameters(*dataset)
        _CACHED_PARAMS[name] = params

    return dataset, params
