# pylint:disable=duplicate-code,logging-fstring-interpolation
"""This script tests an existing model"""
import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional

import git
import torch

from aanntwin.__main__ import DATASET_NAME_DEFAULT
from aanntwin.__main__ import ModelParams
from aanntwin.__main__ import SEED_DEFAULT
from aanntwin.basic import get_device
from aanntwin.basic import test_model
from aanntwin.datasets import get_dataset_and_params
from aanntwin.models import Main
from aanntwin.models import Nonidealities
from aanntwin.models import Normalization
from aanntwin.parser import add_arguments_from_dataclass_fields
from aanntwin.parser import construct_dataclass_from_args


def test_only(
    filepath: Path,
    *,
    dataset_name: str = DATASET_NAME_DEFAULT,
    model_params: Optional[ModelParams] = None,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    seed: Optional[int] = SEED_DEFAULT,
) -> float:
    # pylint:disable=too-many-arguments
    """Test a model using pre-trained parameters"""

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

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

    return accuracy


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("filepath", type=Path)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME_DEFAULT)
    add_arguments_from_dataclass_fields(ModelParams, parser, "model_params")
    add_arguments_from_dataclass_fields(Nonidealities, parser, "nonidealities")
    add_arguments_from_dataclass_fields(Normalization, parser, "normalization")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
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
        model_params=construct_dataclass_from_args(ModelParams, args, "model_params"),
        nonidealities=construct_dataclass_from_args(
            Nonidealities, args, "nonidealities"
        ),
        normalization=construct_dataclass_from_args(
            Normalization, args, "normalization"
        ),
        seed=args.seed,
    )

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
