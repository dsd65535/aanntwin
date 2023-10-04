"""This module provides basic functions"""
from typing import Optional
from typing import Tuple

import torch


def get_device() -> str:
    """Get the correct device to use"""

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def extract_shape(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Size, int]:
    """Extract the tensor shape and number of labels in a dataloader"""

    shape_set = set(tensor.shape for tensor, _ in dataloader.dataset)
    if len(shape_set) < 1:
        raise RuntimeError("Empty Dataloader")
    if len(shape_set) > 1:
        raise RuntimeError("Inconsistent Dataloader")

    label_set = set(label for _, label in dataloader.dataset)

    return shape_set.pop(), len(label_set)


def train_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cpu",
    print_rate: Optional[int] = None,
) -> None:
    # pylint:disable=too-many-arguments
    """Train a model on a dataloader"""

    if print_rate is not None:
        total_size = len(dataloader.dataset)
        print("              Last Loss")

    model.train()

    for idx_batch, (tensor_in, correct_label) in enumerate(dataloader):
        tensor_in = tensor_in.to(device)
        correct_label = correct_label.to(device)

        prediction = model(tensor_in)
        loss = loss_fn(prediction, correct_label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if print_rate is not None and idx_batch % print_rate == print_rate - 1:
            print(
                f"[{(idx_batch + 1) * len(tensor_in):>5d}/{total_size:>5d}]  {loss.item():<9f}"
            )


def test_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Test a model on a dataloader"""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for tensor_in, correct_label in dataloader:
            tensor_in = tensor_in.to(device)
            correct_label = correct_label.to(device)

            prediction = model(tensor_in)
            loss = loss_fn(prediction, correct_label)

            total_loss += loss.item()
            total_correct += (
                (prediction.argmax(1) == correct_label).type(torch.float).sum().item()
            )

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)