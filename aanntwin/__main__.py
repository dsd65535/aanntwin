# pylint:disable=logging-fstring-interpolation
"""This script trains and tests the Main model"""
import argparse
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import git
import numpy as np
import torch

from aanntwin.basic import get_device
from aanntwin.basic import test_model
from aanntwin.basic import train_model
from aanntwin.datasets import get_dataset_and_params
from aanntwin.models import FullModelParams
from aanntwin.models import Main
from aanntwin.models import Nonidealities
from aanntwin.models import Normalization
from aanntwin.normalize import normalize_values
from aanntwin.parser import add_arguments_from_dataclass_fields
from aanntwin.utils import hash_str

MODELCACHEDIR = Path("cache/models")
DATASET_NAME_DEFAULT = "MNIST"
COUNT_EPOCH_DEFAULT = 100
SEED_DEFAULT = 42


@dataclass
class TrainParams:
    """Parameters used during training"""

    batch_size: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.0

    def __str__(self) -> str:
        return f"{self.batch_size}_{self.lr}_{self.weight_decay}_{self.momentum}"


@dataclass
class ModelParams:
    """Dataset-independent parameters for Main model"""

    conv_out_channels: int = 4
    kernel_size: int = 6
    stride: int = 2
    padding: int = 1
    pool_size: int = 1
    additional_layers: Optional[List[int]] = None

    def get_full_model_params(
        self, in_channels: int, in_size: int, feature_count: int
    ) -> FullModelParams:
        """Convert to FullModelParams"""

        return FullModelParams(
            in_size=in_size,
            in_channels=in_channels,
            conv_out_channels=self.conv_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            pool_size=self.pool_size,
            feature_count=feature_count,
            additional_layers=self.additional_layers,
        )

    def __str__(self) -> str:
        return (
            f"{self.conv_out_channels}_{self.kernel_size}_{self.stride}_"
            f"{self.padding}_{self.pool_size}_{self.additional_layers}"
        )


def _get_largest_cached_epoch_number(search_dirpath: Path, basename: str) -> int:
    """Get the largest epoch number cached in a directory"""

    cached_indices = []
    for filepath in search_dirpath.glob("*"):
        match = re.match(rf"^{search_dirpath}/{basename}_(\d+).pth$", str(filepath))
        if match is None:
            continue
        cached_indices.append(int(match.group(1)))

    largest_cached_epoch_number = len(cached_indices)

    if largest_cached_epoch_number > 0:
        if len(set(cached_indices)) != largest_cached_epoch_number:
            raise RuntimeError("Indices not unique")
        if min(cached_indices) != 1:
            raise RuntimeError(
                f"Expected minimum index to be 1, not {min(cached_indices)}"
            )
        if max(cached_indices) != largest_cached_epoch_number:
            raise RuntimeError(
                f"Expected minimum index to be {largest_cached_epoch_number}, "
                f"not {max(cached_indices)}"
            )

    return largest_cached_epoch_number


def _get_epoch_params(
    *,
    idx_epoch: int,
    count_epoch: int,
    largest_cached_epoch_number: int,
    test_each_epoch: bool,
    test_last_epoch: bool,
    record: bool,
) -> Tuple[bool, bool, bool]:
    # pylint:disable=too-many-arguments
    """Get parameters for an epoch"""

    last_epoch = idx_epoch == count_epoch

    train_this_epoch = idx_epoch > largest_cached_epoch_number
    train_next_epoch = (
        idx_epoch + 1 > largest_cached_epoch_number and idx_epoch < count_epoch
    )

    if last_epoch:
        test_this_epoch = test_each_epoch or test_last_epoch or record
    else:
        test_this_epoch = (
            test_each_epoch and logging.getLogger().getEffectiveLevel() >= logging.INFO
        )

    skip_this_epoch = (
        not last_epoch
        and not train_this_epoch
        and not train_next_epoch
        and not test_this_epoch
    )

    return skip_this_epoch, train_this_epoch, test_this_epoch


def train_and_test(
    *,
    dataset_name: str = DATASET_NAME_DEFAULT,
    train_params: Optional[TrainParams] = None,
    model_params: Optional[ModelParams] = None,
    training_nonidealities: Optional[Nonidealities] = None,
    testing_nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    count_epoch: int = COUNT_EPOCH_DEFAULT,
    use_cache: bool = True,
    print_rate: Optional[int] = None,
    test_each_epoch: bool = False,
    test_last_epoch: bool = False,
    record: bool = False,
    seed: Optional[int] = SEED_DEFAULT,
) -> Tuple[
    Tuple[
        torch.nn.Module,
        torch.nn.Module,
        torch.utils.data.DataLoader,
        FullModelParams,
        str,
    ],
    Optional[Tuple[float, float]],
]:
    # pylint:disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    """Train and Test the Main model

    This function is based on:
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    elif use_cache:
        logging.warning("Using cache with a random seed!")

    if model_params is None:
        model_params = ModelParams()
    if train_params is None:
        train_params = TrainParams()
    if training_nonidealities is None:
        training_nonidealities = Nonidealities()
    if testing_nonidealities is None:
        testing_nonidealities = training_nonidealities
    if normalization is None:
        normalization = Normalization()

    device = get_device()

    logging.info("Loading dataset...")
    (train_dataloader, test_dataloader), dataset_params = get_dataset_and_params(
        name=dataset_name, batch_size=train_params.batch_size
    )

    full_model_params = model_params.get_full_model_params(*dataset_params)
    training_model = Main(
        full_model_params,
        training_nonidealities,
        normalization,
        False,
    ).to(device)
    testing_model = Main(
        full_model_params,
        testing_nonidealities,
        normalization,
        record,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        training_model.parameters(),
        lr=train_params.lr,
        weight_decay=train_params.weight_decay,
        momentum=train_params.momentum,
    )

    cache_basename = (
        f"{dataset_name}_{hash_str(str(train_params))}_{model_params}_"
        f"{hash_str(str(training_nonidealities))}_{seed}"
    )
    if use_cache:
        MODELCACHEDIR.mkdir(parents=True, exist_ok=True)
        largest_cached_epoch_number = _get_largest_cached_epoch_number(
            MODELCACHEDIR, cache_basename
        )
    else:
        largest_cached_epoch_number = 0

    result = None
    for idx_epoch in range(1, count_epoch + 1):
        skip_this_epoch, train_this_epoch, test_this_epoch = _get_epoch_params(
            idx_epoch=idx_epoch,
            count_epoch=count_epoch,
            largest_cached_epoch_number=largest_cached_epoch_number,
            test_each_epoch=test_each_epoch,
            record=record,
            test_last_epoch=test_last_epoch,
        )

        if skip_this_epoch:
            continue

        logging.info(f"Epoch {idx_epoch}/{count_epoch}")
        cache_filepath = Path(f"{MODELCACHEDIR}/{cache_basename}_{idx_epoch}.pth")

        if train_this_epoch:
            logging.info("Training...")
            train_model(
                training_model,
                train_dataloader,
                loss_fn,
                optimizer,
                device=device,
                print_rate=print_rate,
            )
            named_state_dict = training_model.named_state_dict()
            if use_cache:
                logging.info(f"Saving to {cache_filepath}...")
                torch.save(named_state_dict, cache_filepath)
        else:
            if not use_cache:
                raise RuntimeError(
                    "Epoch needed but neither training nor cache available"
                )
            logging.info(f"Loading from {cache_filepath}...")
            named_state_dict = torch.load(
                cache_filepath, map_location=torch.device(device)
            )
            training_model.load_named_state_dict(named_state_dict)

        testing_model.load_named_state_dict(named_state_dict)

        if test_this_epoch:
            if record:
                for layer in testing_model.store.values():
                    layer.clear()
            logging.info("Testing...")
            result = test_model(testing_model, test_dataloader, loss_fn, device=device)
            avg_loss, accuracy = result
            logging.info(f"Average Loss:  {avg_loss:<9f}")
            logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

    return (testing_model, loss_fn, test_dataloader, full_model_params, device), result


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME_DEFAULT)
    add_arguments_from_dataclass_fields(TrainParams, parser)
    add_arguments_from_dataclass_fields(ModelParams, parser)
    add_arguments_from_dataclass_fields(Nonidealities, parser, prefix="training")
    add_arguments_from_dataclass_fields(Nonidealities, parser, prefix="testing")
    add_arguments_from_dataclass_fields(Normalization, parser)
    parser.add_argument("--count_epoch", type=int, default=COUNT_EPOCH_DEFAULT)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--print_rate", type=int, nargs="?")
    parser.add_argument("--test_each_epoch", action="store_true")
    parser.add_argument("--test_last_epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--output_path", type=Path, nargs="?")

    return parser.parse_args()


def main() -> None:
    """Main Function"""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Imports done, running script...")

    args = parse_args()

    if args.print_git_info:
        repo = git.Repo(search_parent_directories=True)
        logging.info(f"Git SHA: {repo.head.object.hexsha}")
        diff = repo.git.diff()
        if diff:
            logging.info(repo.git.diff())

    if args.timed:
        start = time.time()

    (testing_model, loss_fn, test_dataloader, _, device), _ = train_and_test(
        dataset_name=args.dataset_name,
        train_params=TrainParams(
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        ),
        model_params=ModelParams(
            conv_out_channels=args.conv_out_channels,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pool_size=args.pool_size,
            additional_layers=args.additional_layers,
        ),
        training_nonidealities=Nonidealities(
            input_noise=args.training_input_noise,
            relu_cutoff=args.training_relu_cutoff,
            relu_out_noise=args.training_relu_out_noise,
            linear_out_noise=args.training_linear_out_noise,
            conv2d_out_noise=args.training_conv2d_out_noise,
            linear_input_clip=args.training_linear_input_clip,
            conv2d_input_clip=args.training_conv2d_input_clip,
        ),
        testing_nonidealities=Nonidealities(
            input_noise=args.testing_input_noise,
            relu_cutoff=args.testing_relu_cutoff,
            relu_out_noise=args.testing_relu_out_noise,
            linear_out_noise=args.testing_linear_out_noise,
            conv2d_out_noise=args.testing_conv2d_out_noise,
            linear_input_clip=args.testing_linear_input_clip,
            conv2d_input_clip=args.testing_conv2d_input_clip,
        ),
        normalization=Normalization(
            min_out=args.min_out,
            max_out=args.max_out,
            min_in=args.min_in,
            max_in=args.max_in,
        ),
        count_epoch=args.count_epoch,
        use_cache=not args.no_cache,
        print_rate=args.print_rate,
        test_each_epoch=args.test_each_epoch,
        test_last_epoch=args.test_last_epoch,
        record=args.normalize,
        seed=args.seed,
    )

    if args.normalize:
        logging.info("Normalizing...")
        normalize_values(testing_model.named_state_dict(), testing_model.store)

        if logging.getLogger().getEffectiveLevel() >= logging.INFO:
            logging.info("Testing...")
            avg_loss, accuracy = test_model(
                testing_model, test_dataloader, loss_fn, device=device
            )
            logging.info(f"Average Loss:  {avg_loss:<9f}")
            logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

    if args.output_path is not None:
        torch.save(testing_model.named_state_dict(), args.output_path)

    if args.timed:
        end = time.time()
        logging.info(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
