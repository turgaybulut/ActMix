import torch
import torch.nn as nn

from src.models.layers.actmix_layer import ActMixLayer


class ActMixMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        basis_functions: list[str] = None,
        dropout_rate: float = 0.0,
        relu_bias: float = 1.5,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [128, 128]
        basis_functions = basis_functions or ["sin", "tanh", "relu", "identity"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.basis_functions = basis_functions
        self.relu_bias = relu_bias

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(ActMixLayer(hidden_dim, basis_functions, relu_bias))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.actmix_layers = [
            layer for layer in self.network if isinstance(layer, ActMixLayer)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def set_temperature(self, temperature: float) -> None:
        for layer in self.actmix_layers:
            layer.set_temperature(temperature)

    def compute_total_entropy(self) -> torch.Tensor:
        entropies = [layer.compute_entropy() for layer in self.actmix_layers]
        return torch.stack(entropies).mean()

    def get_all_mixing_coefficients(self) -> list[torch.Tensor]:
        return [layer.get_mixing_coefficients() for layer in self.actmix_layers]
