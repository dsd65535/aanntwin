"""Normalization functions"""
from typing import Dict
from typing import List
from typing import Optional

import torch


def normalize_values(
    named_state_dict: Dict[str, torch.Tensor],
    store: Dict[str, List[torch.Tensor]],
    new_refs: Optional[Dict[str, float]] = None,
    stdevs: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize all intermediate values"""

    if new_refs is None:
        new_refs = {}
    if stdevs is None:
        stdevs = {}

    current_normalization = 1.0
    for layer, tensors in store.items():
        if (
            f"{layer}_bias" not in named_state_dict
            or f"{layer}_weight" not in named_state_dict
        ):
            continue
        values = [value for tensor in tensors for value in tensor.flatten().tolist()]

        stdev = stdevs.get(layer)

        if stdev is None:
            old_ref = max(values)
        else:
            raise NotImplementedError

        normalization = old_ref / new_refs.get("layer", 1.0)

        named_state_dict[f"{layer}_bias"] /= normalization
        named_state_dict[f"{layer}_weight"] /= normalization / current_normalization

        current_normalization *= normalization

    return named_state_dict
