# pylint:disable=logging-fstring-interpolation,duplicate-code
"""This script determines the effect of noise"""
import argparse
import json
import logging
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import git
import numpy as np
import torch

from aanntwin.__main__ import COUNT_EPOCH_DEFAULT
from aanntwin.__main__ import DATASET_NAME_DEFAULT
from aanntwin.__main__ import ModelParams
from aanntwin.__main__ import SEED_DEFAULT
from aanntwin.__main__ import train_and_test
from aanntwin.__main__ import TrainParams
from aanntwin.basic import test_model
from aanntwin.models import Main
from aanntwin.models import Nonidealities
from aanntwin.models import Normalization
from aanntwin.normalize import normalize_values
from aanntwin.parser import add_arguments_from_dataclass_fields


def run(
    noises_train: List[Optional[float]],
    noises_test: List[Optional[float]],
    output_filepath: Path,
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
    seed: Optional[int] = SEED_DEFAULT,
    noise_type: str = "input",
    runs_per_point: int = 1,
) -> None:
    # pylint:disable=too-many-arguments,too-many-locals
    """Run"""

    if training_nonidealities is None:
        training_nonidealities = Nonidealities()
    if testing_nonidealities is None:
        testing_nonidealities = training_nonidealities

    results: Dict[
        Optional[float], Dict[Optional[float], List[Tuple[float, float]]]
    ] = {}
    for noise_train in noises_train:
        results[noise_train] = {}
        (
            testing_model,
            loss_fn,
            test_dataloader,
            full_model_params,
            device,
        ), _ = train_and_test(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=model_params,
            training_nonidealities=replace(
                training_nonidealities, input_noise=noise_train
            ),
            testing_nonidealities=testing_nonidealities,
            normalization=normalization,
            count_epoch=count_epoch,
            use_cache=use_cache,
            print_rate=print_rate,
            test_each_epoch=test_each_epoch,
            test_last_epoch=False,
            record=True,
            seed=seed,
        )
        normalize_values(testing_model.named_state_dict(), testing_model.store)
        for noise_test in noises_test:
            if seed is not None:
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
            noisy_testing_model = Main(
                full_model_params,
                replace(
                    testing_nonidealities,
                    **{f"{noise_type}_noise": noise_test},  # type:ignore
                ),
                normalization,
            ).to(device)
            noisy_testing_model.load_named_state_dict(testing_model.named_state_dict())
            results[noise_train][noise_test] = []
            for idx_run in range(runs_per_point):
                result = test_model(
                    noisy_testing_model, test_dataloader, loss_fn, device=device
                )
                logging.info(f"{noise_train} {noise_test} {idx_run} {result}")
                results[noise_train][noise_test].append(result)
                with output_filepath.open("w") as output_file:
                    json.dump(results, output_file, indent=4)


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("output_filepath", type=Path)
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
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--output_path", type=Path, nargs="?")
    parser.add_argument("--noise_type", type=str, default="input")
    parser.add_argument("--runs_per_point", type=int, default=1)

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

    run(
        [None] + [noise_idx / 10 for noise_idx in range(11)],
        [None] + [noise_idx / 100 for noise_idx in range(101)],
        args.output_filepath,
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
            linear_input_nonlin=args.training_linear_input_nonlin,
            conv2d_input_nonlin=args.training_conv2d_input_nonlin,
        ),
        testing_nonidealities=Nonidealities(
            input_noise=args.testing_input_noise,
            relu_cutoff=args.testing_relu_cutoff,
            relu_out_noise=args.testing_relu_out_noise,
            linear_out_noise=args.testing_linear_out_noise,
            conv2d_out_noise=args.testing_conv2d_out_noise,
            linear_input_clip=args.testing_linear_input_clip,
            conv2d_input_clip=args.testing_conv2d_input_clip,
            linear_input_nonlin=args.testing_linear_input_nonlin,
            conv2d_input_nonlin=args.testing_conv2d_input_nonlin,
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
        seed=args.seed,
        noise_type=args.noise_type,
        runs_per_point=args.runs_per_point,
    )

    if args.timed:
        end = time.time()
        logging.info(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
