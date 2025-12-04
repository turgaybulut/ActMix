import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        activation: nn.Module = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [128, 128]
        activation = activation or nn.ReLU()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReLUMLP(BaselineMLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=nn.ReLU(),
            dropout_rate=dropout_rate,
        )


class GeLUMLP(BaselineMLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=nn.GELU(),
            dropout_rate=dropout_rate,
        )


class PReLUMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [128, 128]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU(num_parameters=hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
