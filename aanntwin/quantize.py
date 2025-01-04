"""Quantization functions"""
from statistics import stdev

import torch


def quantize_value(
    value: float, ref_value: float, bit_count: int, with_zero: bool = False
) -> float:
    """Quantize a value to a number of bits not counting sign"""

    step = ref_value / 2**bit_count

    if with_zero:
        quantized_value = round(value / step) * step
        return max(min(quantized_value, ref_value - step), -ref_value + step)

    quantized_value = (round(value / step + 0.5) - 0.5) * step
    return max(min(quantized_value, ref_value - step / 2), -ref_value + step / 2)


def quantize_values_ref(
    values: torch.Tensor, bit_count: int, ref_value: float, with_zero: bool = False
) -> torch.Tensor:
    """Quantize all values in a Tensor to a number of bits not counting sign
    using a reference value
    """

    device = values.device
    return (
        values.to("cpu")
        .apply_(lambda value: quantize_value(value, ref_value, bit_count, with_zero))
        .to(device)
    )


def quantize_values_stdev(
    values: torch.Tensor, bit_count: int, stdev_count: float, with_zero: bool = False
) -> torch.Tensor:
    """Quantize all values in a Tensor to a number of bits not counting sign
    using a reference value
    """

    ref_value = stdev(values.flatten().tolist()) * stdev_count
    return quantize_values_ref(values, bit_count, ref_value, with_zero)
