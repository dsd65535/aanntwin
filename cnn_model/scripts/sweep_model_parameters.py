# pylint:disable=duplicate-code
"""This script sweeps the model parameters"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import git

from cnn_model.__main__ import ModelParams
from cnn_model.__main__ import train_and_test
from cnn_model.__main__ import TrainParams
from cnn_model.basic import test_model
from cnn_model.models import Nonidealities
from cnn_model.models import Normalization
from cnn_model.parser import add_arguments_from_dataclass_fields


def generate_model_params() -> List[ModelParams]:
    # pylint:disable=too-many-nested-blocks
    """Generate the set of ModelParams"""

    model_params_list = []
    for conv_out_channels_exp in range(8):
        conv_out_channels = 2**conv_out_channels_exp
        for kernel_size in range(1, 8):
            for stride in range(1, 8):
                for padding in range(kernel_size):
                    for pool_size in range(1, 8):
                        for additional_layers in [None]:
                            model_params_list.append(
                                ModelParams(
                                    conv_out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    pool_size,
                                    additional_layers,
                                )
                            )

    return model_params_list


def run(
    *,
    dataset_name: str,
    train_params: Optional[TrainParams],
    model_params: ModelParams,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    use_cache: bool = True,
    retrain: bool = False,
    print_rate: Optional[int] = None,
) -> Optional[float]:
    # pylint:disable=too-many-arguments
    """Run a test"""

    try:
        model, loss_fn, test_dataloader, device = train_and_test(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=model_params,
            nonidealities=nonidealities,
            normalization=normalization,
            use_cache=use_cache,
            retrain=retrain,
            print_rate=print_rate,
        )
    except ValueError:
        logging.exception("Training failed")
        return None

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)

    return accuracy


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("database_filepath", type=Path, nargs="?")
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    add_arguments_from_dataclass_fields(TrainParams, parser)
    add_arguments_from_dataclass_fields(Nonidealities, parser)
    add_arguments_from_dataclass_fields(Normalization, parser)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--print_rate", type=int, nargs="?")
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Main Function"""

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

    train_params = TrainParams(
        count_epoch=args.count_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_train=args.noise_train,
    )
    nonidealities = Nonidealities(
        relu_cutoff=args.relu_cutoff,
        relu_out_noise=args.relu_out_noise,
        linear_out_noise=args.linear_out_noise,
    )
    normalization = Normalization(
        min_out=args.min_out,
        max_out=args.max_out,
        min_in=args.min_in,
        max_in=args.max_in,
    )

    if args.database_filepath is None:
        database: Optional[Dict[str, Optional[float]]] = None
    elif args.database_filepath.exists():
        with args.database_filepath.open("r", encoding="UTF-8") as database_file:
            database = json.load(database_file)
    else:
        database = {}

    model_params_list = generate_model_params()
    print(
        "conv_out_channels,kernel_size,stride,padding,pool_size,additional_layers,accuracy"
    )
    for model_params in model_params_list:
        if database is not None and str(model_params) in database:
            accuracy = database[str(model_params)]
        else:
            accuracy = run(
                dataset_name=args.dataset_name,
                train_params=train_params,
                model_params=model_params,
                nonidealities=nonidealities,
                normalization=normalization,
                use_cache=not args.no_cache,
                retrain=args.retrain,
                print_rate=args.print_rate,
            )
            if database is not None:
                database[str(model_params)] = accuracy
                if args.database_filepath is None:
                    raise RuntimeError
                with args.database_filepath.open(
                    "w", encoding="UTF-8"
                ) as database_file:
                    json.dump(database, database_file, indent=4)

        accuracy_str = "N/A" if accuracy is None else f"{accuracy*100}%"
        print(
            f"{model_params.conv_out_channels},"
            f"{model_params.kernel_size},"
            f"{model_params.stride},"
            f"{model_params.padding},"
            f"{model_params.pool_size},"
            f"{model_params.additional_layers},"
            f"{accuracy_str}"
        )

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
