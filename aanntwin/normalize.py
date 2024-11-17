"""Normalization functions"""
from typing import Dict
from typing import List

import torch


def normalize_values(
    named_state_dict: Dict[str, torch.Tensor], store: Dict[str, List[torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Normalize all intermediate values to a maximum of 1.0"""

    max_conv2d = max(
        val for tensor in store["conv2d"] for val in tensor.flatten().tolist()
    )
    max_linear = max(
        val for tensor in store["linear"] for val in tensor.flatten().tolist()
    )

    named_state_dict["conv2d_bias"] /= max_conv2d
    named_state_dict["conv2d_weight"] /= max_conv2d
    named_state_dict["linear_bias"] /= max_linear
    named_state_dict["linear_weight"] /= max_linear / max_conv2d

    return named_state_dict
