from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class TabularModel(Protocol):
    input_dim: int
    output_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class MixableModel(Protocol):
    def set_temperature(self, temperature: float) -> None: ...
    def compute_total_entropy(self) -> torch.Tensor: ...
    def get_all_mixing_coefficients(self) -> list[torch.Tensor]: ...


class BaseMLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(in_features, out_features),
            activation,
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


def create_activation(name: str, num_features: int | None = None) -> nn.Module:
    if name == "prelu":
        assert num_features is not None
        return nn.PReLU(num_parameters=num_features)

    activation_class = ACTIVATION_REGISTRY.get(name)
    assert activation_class is not None, f"Unknown activation: {name}"
    return activation_class()


class StaticMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [128, 128]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            act = create_activation(activation, hidden_dim)
            layers.append(BaseMLPBlock(prev_dim, hidden_dim, act, dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp_relu": StaticMLP,
    "mlp_gelu": StaticMLP,
    "mlp_prelu": StaticMLP,
}


def get_activation_for_model(model_name: str) -> str:
    mapping = {
        "mlp_relu": "relu",
        "mlp_gelu": "gelu",
        "mlp_prelu": "prelu",
    }
    return mapping.get(model_name, "relu")
