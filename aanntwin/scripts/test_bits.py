"""This script test the effects of quantization on accuracy"""
from statistics import stdev
from typing import Optional

import numpy as np

from aanntwin.__main__ import ModelParams
from aanntwin.__main__ import train_and_test
from aanntwin.__main__ import TrainParams
from aanntwin.basic import test_model
from aanntwin.models import Nonidealities
from aanntwin.models import Normalization
from aanntwin.quantize import quantize_values_ref


def main(
    min_bits: int = 0,
    max_bits: int = 7,
    min_ref: float = 0.1,
    max_ref: float = 3.0,
    count_ref: int = 30,
    look_at_linear_layer: bool = True,
    look_at_bias: bool = False,
    dataset_name: str = "MNIST",
    train_params: Optional[TrainParams] = None,
    model_params: Optional[ModelParams] = None,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    count_epoch: int = 5,
    use_cache: bool = True,
    print_rate: Optional[int] = None,
) -> None:
    # pylint:disable=too-many-locals,duplicate-code,too-many-arguments
    """Test the effects of quantization"""

    layer_params = (
        ("linear_bias" if look_at_bias else "linear_weight")
        if look_at_linear_layer
        else ("conv2d_bias" if look_at_bias else "conv2d_weight")
    )

    if model_params is None:
        model_params = ModelParams()

    model, loss_fn, test_dataloader, device = train_and_test(
        dataset_name=dataset_name,
        train_params=train_params,
        model_params=model_params,
        nonidealities=nonidealities,
        normalization=normalization,
        count_epoch=count_epoch,
        use_cache=use_cache,
        print_rate=print_rate,
    )

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"Unlimited Accuracy: {accuracy}")

    original_state_dict = {k: v.clone() for k, v in model.named_state_dict().items()}
    weight_stdev = stdev(original_state_dict[layer_params].flatten().tolist())

    print(
        "ref,float,"
        + ",".join(str(count_bits) for count_bits in range(min_bits, max_bits + 1))
    )

    for ref in np.linspace(min_ref * weight_stdev, max_ref * weight_stdev, count_ref):
        # pylint:disable=cell-var-from-loop
        print(ref, end=",")

        limited_state_dict = {k: v.clone() for k, v in original_state_dict.items()}
        limited_state_dict[layer_params] = (
            limited_state_dict[layer_params]
            .to("cpu")
            .apply_(lambda val: min(ref, max(-ref, val)))
            .to(device)
        )
        model.load_named_state_dict(limited_state_dict)
        _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
        print(accuracy, end=",")

        for count_bits in range(min_bits, max_bits + 1):
            new_state_dict = {k: v.clone() for k, v in limited_state_dict.items()}
            new_state_dict[layer_params] = quantize_values_ref(
                new_state_dict[layer_params], count_bits, ref
            )
            model.load_named_state_dict(new_state_dict)
            _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
            print(accuracy, end=",")

        print()


if __name__ == "__main__":
    main()
