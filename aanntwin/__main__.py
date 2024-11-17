# pylint:disable=logging-fstring-interpolation
"""This script trains and tests the Main model"""
import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import git
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

MODELCACHEDIR = Path("cache/models")


@dataclass
class TrainParams:
    """Parameters used during training"""

    batch_size: int = 1
    lr: float = 1e-3
    noise_train: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.batch_size}_{self.lr}_{self.noise_train}"


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


def get_largest_cached_epoch_number(search_dirpath: Path, basename: str) -> int:
    """Get the largest epoch number cached in a directory"""
    cached_indices = []
    for filepath in search_dirpath.glob("*"):
        match = re.match(rf"{search_dirpath}/{basename}_(\d+).pth", str(filepath))
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


def train_and_test(
    dataset_name: str = "MNIST",
    train_params: Optional[TrainParams] = None,
    model_params: Optional[ModelParams] = None,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    count_epoch: int = 5,
    use_cache: bool = True,
    print_rate: Optional[int] = None,
    test_each_epoch: bool = False,
    record: bool = False,
    normalize: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.utils.data.DataLoader, str]:
    # pylint:disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    """Train and Test the Main model

    This function is based on:
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """

    if model_params is None:
        model_params = ModelParams()
    if train_params is None:
        train_params = TrainParams()
    if nonidealities is None:
        nonidealities = Nonidealities()
    if normalization is None:
        normalization = Normalization()

    device = get_device()
    logging_info = logging.getLogger().getEffectiveLevel() >= logging.INFO

    logging.info("Loading dataset...")
    (train_dataloader, test_dataloader), dataset_params = get_dataset_and_params(
        name=dataset_name, batch_size=train_params.batch_size
    )

    model = Main(
        model_params.get_full_model_params(*dataset_params),
        nonidealities,
        normalization,
        record or normalize,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params.lr)

    cache_basename = f"{dataset_name}_{train_params}_{model_params}_{nonidealities}"
    if use_cache:
        MODELCACHEDIR.mkdir(parents=True, exist_ok=True)
        largest_cached_epoch_number = get_largest_cached_epoch_number(
            MODELCACHEDIR, cache_basename
        )
    else:
        largest_cached_epoch_number = 0

    for idx_epoch in range(count_epoch):
        last_epoch = idx_epoch == count_epoch - 1
        train_this_epoch = idx_epoch >= largest_cached_epoch_number
        train_next_epoch = idx_epoch >= largest_cached_epoch_number - 1
        test_this_epoch = (logging_info and (test_each_epoch or last_epoch)) or (
            normalize and last_epoch
        )

        if not train_this_epoch and not train_next_epoch and not test_this_epoch:
            continue

        logging.info(f"Epoch {idx_epoch+1}/{count_epoch}")
        cache_filepath = Path(f"{MODELCACHEDIR}/{cache_basename}_{idx_epoch+1}.pth")

        if train_this_epoch:
            logging.info("Training...")
            train_model(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device=device,
                print_rate=print_rate,
                noise=train_params.noise_train,
            )
            if use_cache:
                logging.info(f"Saving to {cache_filepath}...")
                torch.save(model.named_state_dict(), cache_filepath)
        else:
            logging.info(f"Loading from {cache_filepath}...")
            named_state_dict = torch.load(cache_filepath)
            model.load_named_state_dict(named_state_dict)

        if test_this_epoch:
            if normalize and last_epoch:
                for layer in model.store.values():
                    layer.clear()
            logging.info("Testing...")
            avg_loss, accuracy = test_model(
                model, test_dataloader, loss_fn, device=device
            )
            logging.info(f"Average Loss:  {avg_loss:<9f}")
            logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

    if normalize:
        logging.info("Normalizing...")
        normalize_values(model.named_state_dict(), model.store)

        if use_cache:
            cache_filepath = Path(
                f"{MODELCACHEDIR}/{cache_basename}_{count_epoch}_norm.pth"
            )
            torch.save(model.named_state_dict(), cache_filepath)

        if logging_info:
            logging.info("Testing...")
            avg_loss, accuracy = test_model(
                model, test_dataloader, loss_fn, device=device
            )
            logging.info(f"Average Loss:  {avg_loss:<9f}")
            logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

    return model, loss_fn, test_dataloader, device


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_name", type=str, default="MNIST")
    add_arguments_from_dataclass_fields(TrainParams, parser)
    add_arguments_from_dataclass_fields(ModelParams, parser)
    add_arguments_from_dataclass_fields(Nonidealities, parser)
    add_arguments_from_dataclass_fields(Normalization, parser)
    parser.add_argument("--count_epoch", type=int, default=5)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--print_rate", type=int, nargs="?")
    parser.add_argument("--test_each_epoch", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")

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

    train_and_test(
        dataset_name=args.dataset_name,
        train_params=TrainParams(
            batch_size=args.batch_size,
            lr=args.lr,
            noise_train=args.noise_train,
        ),
        model_params=ModelParams(
            conv_out_channels=args.conv_out_channels,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pool_size=args.pool_size,
            additional_layers=args.additional_layers,
        ),
        nonidealities=Nonidealities(
            relu_cutoff=args.relu_cutoff,
            relu_out_noise=args.relu_out_noise,
            linear_out_noise=args.linear_out_noise,
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
        normalize=args.normalize,
    )

    if args.timed:
        end = time.time()
        logging.info(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
