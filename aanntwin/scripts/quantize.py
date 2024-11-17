"""This script quantizes an existing model"""
import argparse
from pathlib import Path

import torch

from aanntwin.quantize import quantize_values_stdev


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("conv2d_weight_bits", type=int)
    parser.add_argument("conv2d_weight_ref", type=float)
    parser.add_argument("linear_weight_bits", type=int)
    parser.add_argument("linear_weight_ref", type=float)

    return parser.parse_args()


def main() -> None:
    """CLI Entry Point"""

    args = parse_args()

    named_state_dict = torch.load(args.input_path)
    named_state_dict["conv2d_weight"] = quantize_values_stdev(
        named_state_dict["conv2d_weight"],
        args.conv2d_weight_bits,
        args.conv2d_weight_ref,
    )
    named_state_dict["linear_weight"] = quantize_values_stdev(
        named_state_dict["linear_weight"],
        args.linear_weight_bits,
        args.linear_weight_ref,
    )
    torch.save(named_state_dict, args.output_path)


if __name__ == "__main__":
    main()
