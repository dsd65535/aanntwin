"""This script analyzes the weights in an existing model"""
import argparse
import logging
from pathlib import Path
from statistics import mean
from statistics import median
from statistics import stdev

import torch


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input_path", type=Path)

    return parser.parse_args()


def _print_header() -> None:
    print(
        "name",
        "\t\tmin",
        "\t\tmean",
        "\t\tmedian",
        "\t\tmax",
        "\t\tstdev",
        "\t\tsum",
        "\t\tlen",
        "\tunique",
    )


def _print_stats(name: str, values: torch.Tensor) -> None:
    abs_values = [abs(value) for value in values.flatten().tolist()]
    print(
        f"{name}"
        f"\t{min(abs_values):.2E}"
        f"\t{mean(abs_values):.2E}"
        f"\t{median(abs_values):.2E}"
        f"\t{max(abs_values):.2E}"
        f"\t{stdev(abs_values):.2E}"
        f"\t{sum(abs_values):.2E}"
        f"\t{len(abs_values)}"
        f"\t{len(set(abs_values))}"
    )


def main() -> None:
    """CLI Entry Point"""

    args = parse_args()

    named_state_dict = torch.load(args.input_path)
    _print_header()
    for name, values in named_state_dict.items():
        name_split = name.split("_")
        value_type = name_split[-1]
        if value_type == "bias":
            continue
        if value_type != "weight":
            logging.error("Unknown value type: %s", value_type)
        name = "_".join(name_split[:-1])
        _print_stats(f"{name}\ttot.", values)
        for idx, subvalues in enumerate(values):
            _print_stats(f"\t{idx}", subvalues)


if __name__ == "__main__":
    main()
