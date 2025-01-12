# pylint:disable=duplicate-code,logging-fstring-interpolation
"""This script tests an existing model"""
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import git
import torch

from aanntwin.__main__ import DATASET_NAME_DEFAULT
from aanntwin.__main__ import ModelParams
from aanntwin.basic import get_device
from aanntwin.basic import test_model
from aanntwin.datasets import get_dataset_and_params
from aanntwin.models import Main
from aanntwin.models import Nonidealities
from aanntwin.models import Normalization
from aanntwin.parser import add_arguments_from_dataclass_fields


def test_only(
    filepath: Path,
    *,
    dataset_name: str = DATASET_NAME_DEFAULT,
    model_params: Optional[ModelParams] = None,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
) -> None:
    """Test a model using pre-trained parameters"""

    if model_params is None:
        model_params = ModelParams()
    if nonidealities is None:
        nonidealities = Nonidealities()
    if normalization is None:
        normalization = Normalization()

    device = get_device()

    logging.info("Loading dataset...")
    (_, test_dataloader), dataset_params = get_dataset_and_params(name=dataset_name)

    model = Main(
        model_params.get_full_model_params(*dataset_params),
        nonidealities,
        normalization,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    logging.info(f"Loading from {filepath}...")
    named_state_dict = torch.load(filepath, map_location=torch.device(device))
    model.load_named_state_dict(named_state_dict)

    logging.info("Testing...")
    avg_loss, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"Average Loss:  {avg_loss:<9f}")
    print(f"Accuracy:      {(100*accuracy):<0.4f}%")


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("filepath", type=Path)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME_DEFAULT)
    add_arguments_from_dataclass_fields(ModelParams, parser)
    add_arguments_from_dataclass_fields(Nonidealities, parser)
    add_arguments_from_dataclass_fields(Normalization, parser)
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")

    return parser.parse_args()


def main() -> None:
    """CLI Entry Point"""

    args = parse_args()

    if args.print_git_info:
        repo = git.Repo(search_parent_directories=True)
        print(f"Git SHA: {repo.head.object.hexsha}")
        diff = repo.git.diff()
        if diff:
            print(repo.git.diff())
        print()

    if args.timed:
        start = time.time()

    test_only(
        args.filepath,
        dataset_name=args.dataset_name,
        model_params=ModelParams(
            conv_out_channels=args.conv_out_channels,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pool_size=args.pool_size,
            additional_layers=args.additional_layers,
        ),
        nonidealities=Nonidealities(
            input_noise=args.input_noise,
            relu_cutoff=args.relu_cutoff,
            relu_mult_out_noise=args.relu_mult_out_noise,
            linear_mult_out_noise=args.linear_mult_out_noise,
        ),
        normalization=Normalization(
            min_out=args.min_out,
            max_out=args.max_out,
            min_in=args.min_in,
            max_in=args.max_in,
        ),
    )

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
