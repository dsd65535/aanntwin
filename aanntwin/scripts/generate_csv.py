# pylint:disable=duplicate-code
"""This script sweeps the full_model_params parameters"""
import argparse
import time

import git

from aanntwin.__main__ import ModelParams
from aanntwin.datasets import get_dataset_and_params


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")

    return parser.parse_args()


def main() -> None:
    # pylint:disable=too-many-nested-blocks
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

    _, dataset_params = get_dataset_and_params(name=args.dataset_name)

    print(
        "conv_out_channels,"
        "kernel_size,"
        "stride,"
        "padding,"
        "pool_size,"
        "in_size,"
        "feature_count,"
        "conv_out_size,"
        "pool_out_size,"
        "final_size,"
    )
    for conv_out_channels_exp in range(8):
        conv_out_channels = 2**conv_out_channels_exp
        for kernel_size in range(1, 8):
            for stride in range(1, 8):
                for padding in range(kernel_size):
                    for pool_size in range(1, 8):
                        try:
                            full_model_params = ModelParams(
                                conv_out_channels,
                                kernel_size,
                                stride,
                                padding,
                                pool_size,
                            ).get_full_model_params(*dataset_params)
                        except ValueError:
                            continue
                        if full_model_params.in_channels != 1:
                            raise NotImplementedError(
                                "Multiple input channels not supported"
                            )
                        if (
                            full_model_params.additional_layers is not None
                            or len(full_model_params.additional_layer_sizes) != 0
                        ):
                            raise NotImplementedError("Additional Layers not supported")
                        print(
                            f"{full_model_params.conv_out_channels},"
                            f"{full_model_params.kernel_size},"
                            f"{full_model_params.stride},"
                            f"{full_model_params.padding},"
                            f"{full_model_params.pool_size},"
                            f"{full_model_params.in_size},"
                            f"{full_model_params.feature_count},"
                            f"{full_model_params.conv_out_size},"
                            f"{full_model_params.pool_out_size},"
                            f"{full_model_params.final_size},"
                        )

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
