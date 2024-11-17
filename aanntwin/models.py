"""This module defines pyTorch modules and layers"""
import math
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

WEIGHT_RE = re.compile(r"layers\.(\d+)\.weight")
BIAS_RE = re.compile(r"layers\.(\d+)\.bias")


@dataclass
class FullModelParams:
    # pylint:disable=too-many-instance-attributes  # dataclass
    """Parameters for Main model"""

    in_size: int
    in_channels: int
    conv_out_channels: int
    kernel_size: int
    stride: int
    padding: int
    pool_size: int
    feature_count: int
    additional_layers: Optional[List[int]]

    conv_out_size: int = field(init=False)
    pool_out_size: int = field(init=False)
    final_size: int = field(init=False)
    additional_layer_sizes: List[Tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.conv_out_size, rem = divmod(
            self.in_size + self.stride + 2 * self.padding - self.kernel_size,
            self.stride,
        )
        if rem:
            raise ValueError("Invalid Convolution Output Size")

        self.pool_out_size, rem = divmod(self.conv_out_size, self.pool_size)
        if rem:
            raise ValueError("Invalid Pool Output Size")

        current_size = self.pool_out_size**2 * self.conv_out_channels
        additional_layer_sizes = []
        if self.additional_layers is not None:
            for next_size in self.additional_layers:
                additional_layer_sizes.append((current_size, next_size))
                current_size = next_size

        self.final_size = current_size
        self.additional_layer_sizes = additional_layer_sizes

    @property
    def mac_counts(self) -> List[int]:
        """Number of MAC structures in each layer"""

        return (
            []
            if self.additional_layer_sizes is None
            else [in_size for in_size, _ in self.additional_layer_sizes]
            + [self.final_size, self.feature_count]
        )

    @property
    def mac_count(self) -> int:
        """Total number of MAC structures"""

        return sum(self.mac_counts)

    @property
    def multiplier_count(self) -> int:
        """Number of Multipliers"""

        current_size = self.kernel_size**2
        multiplier_count = 0
        for mac_count in self.mac_counts:
            multiplier_count += current_size * mac_count
            current_size = mac_count

        return multiplier_count


@dataclass
class Nonidealities:
    """Nonidealities parameters for Main model"""

    relu_cutoff: float = 0.0
    relu_out_noise: Optional[float] = None
    linear_out_noise: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.relu_cutoff}_{self.relu_out_noise}_{self.linear_out_noise}"


@dataclass
class Normalization:
    """Normalization parameters for Main model"""

    min_out: float = 0.0
    max_out: float = 1.0
    min_in: float = 0.0
    max_in: float = 1.0


class Recorder(torch.nn.Module):
    """Layer that records values"""

    def __init__(self, store: List[torch.Tensor]) -> None:
        super().__init__()
        self.store = store

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        self.store.append(x)
        return x


class Normalize(torch.nn.Module):
    """Layer that converts to a realistic voltage"""

    def __init__(
        self, min_out: float, max_out: float, min_in: float = 0.0, max_in: float = 1.0
    ) -> None:
        super().__init__()

        self.slope = torch.Tensor([(max_out - min_out) / (max_in - min_in)])
        self.offset = torch.Tensor([min_out - self.slope * min_in])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        return torch.add(
            self.offset.to(x.device), torch.multiply(self.slope.to(x.device), x)
        )


class ReLU(torch.nn.Module):
    """Re-implementation of ReLU"""

    def __init__(self, cutoff: float = 0.0, out_noise: Optional[float] = None) -> None:
        super().__init__()

        self.cutoff = torch.tensor([cutoff])

        self.out_noise = out_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        out = torch.max(self.cutoff.to(x.device), x)
        if self.out_noise is not None:
            out = torch.add(out, self.out_noise * torch.randn(out.shape).to(out.device))

        return out


class Linear(torch.nn.Module):
    """Re-implementation of Linear"""

    def __init__(
        self, in_features: int, out_features: int, out_noise: Optional[float] = None
    ) -> None:
        super().__init__()

        self.in_features, self.out_features = in_features, out_features

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self.out_noise = out_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        out = torch.add(torch.mm(x, self.weight.t()), self.bias)
        if self.out_noise is not None:
            out = torch.add(out, self.out_noise * torch.randn(out.shape).to(out.device))

        return out


class Main(torch.nn.Module):
    """A configurable CNN model similar to the one presented in:

    Nikita Mirchandani. Ultra-Low Power and Robust Analog Computing
    Circuits and System Design Framework for Machine Learning Applications.
    """

    def __init__(
        self,
        full_model_params: FullModelParams,
        nonidealities: Optional[Nonidealities] = None,
        normalization: Optional[Normalization] = None,
        record: bool = False,
    ) -> None:
        super().__init__()

        if nonidealities is None:
            nonidealities = Nonidealities()

        if record:
            self.store: Dict[str, List[torch.Tensor]] = {}

        layers: List[Tuple[torch.nn.Module, str]] = []

        if normalization is not None:
            layers.append(
                (
                    Normalize(
                        normalization.min_out,
                        normalization.max_out,
                        normalization.min_in,
                        normalization.max_in,
                    ),
                    "normalize",
                )
            )
        if record:
            self.store["input"] = []
            layers.append(((Recorder(self.store["input"]), "input_record")))

        layers.append(
            (
                torch.nn.Conv2d(
                    full_model_params.in_channels,
                    full_model_params.conv_out_channels,
                    full_model_params.kernel_size,
                    full_model_params.stride,
                    full_model_params.padding,
                ),
                "conv2d",
            )
        )
        if record:
            self.store["conv2d"] = []
            layers.append(((Recorder(self.store["conv2d"]), "conv2d_record")))
        layers.append(
            (
                ReLU(nonidealities.relu_cutoff, nonidealities.relu_out_noise),
                "conv2d_relu",
            )
        )
        layers.append((torch.nn.MaxPool2d(full_model_params.pool_size), "maxpool2d"))
        layers.append((torch.nn.Flatten(), "flatten"))

        for idx, (in_size, out_size) in enumerate(
            full_model_params.additional_layer_sizes
        ):
            name = f"additional_linear_{idx}"
            layers.append((Linear(in_size, out_size), name))
            if record:
                self.store[name] = []
                layers.append((Recorder(self.store[name]), f"{name}_record"))
            layers.append((ReLU(), f"{name}_relu"))

        layers.append(
            (
                Linear(
                    full_model_params.final_size,
                    full_model_params.feature_count,
                    nonidealities.linear_out_noise,
                ),
                "linear",
            )
        )
        if record:
            self.store["linear"] = []
            layers.append(((Recorder(self.store["linear"]), "linear_record")))

        self.layers = torch.nn.Sequential(*(layer for layer, _ in layers))
        self.layer_names = [name for _, name in layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Total forward computation"""

        return self.layers(x)

    def _named_key_from_idx_key(self, idx_key: str) -> str:
        """Convert an indexed key to a named key"""

        match = WEIGHT_RE.match(idx_key)
        if match is not None:
            idx = int(match.group(1))
            return f"{self.layer_names[idx]}_weight"

        match = BIAS_RE.match(idx_key)
        if match is not None:
            idx = int(match.group(1))
            return f"{self.layer_names[idx]}_bias"

        raise ValueError(f"Cannot parse indexed key: {idx_key}")

    def named_state_dict(self) -> Dict:
        """Return a state dict using names instead of layer numbers"""

        return {
            self._named_key_from_idx_key(idx_key): value
            for idx_key, value in self.state_dict().items()
        }

    def load_named_state_dict(self, named_state_dict: Dict) -> None:
        """Load a state dict using names instead of layer numbers"""

        state_dict = {}
        for idx, name in enumerate(self.layer_names):
            if f"{name}_weight" in named_state_dict:
                state_dict[f"layers.{idx}.weight"] = named_state_dict[f"{name}_weight"]
            if f"{name}_bias" in named_state_dict:
                state_dict[f"layers.{idx}.bias"] = named_state_dict[f"{name}_bias"]

        self.load_state_dict(state_dict)
