from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActMixLayer(nn.Module):
    BASIS_FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "sin": torch.sin,
        "tanh": torch.tanh,
        "relu": F.relu,
        "identity": lambda x: x,
    }

    def __init__(
        self,
        num_features: int,
        basis_functions: list[str] = None,
        relu_bias: float = 1.5,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.basis_function_names = basis_functions or [
            "sin",
            "tanh",
            "relu",
            "identity",
        ]
        self.num_basis = len(self.basis_function_names)

        assert all(name in self.BASIS_FUNCTIONS for name in self.basis_function_names)

        self.basis_fns = [
            self.BASIS_FUNCTIONS[name] for name in self.basis_function_names
        ]

        self.mixing_weights = nn.Parameter(torch.zeros(num_features, self.num_basis))
        self._initialize_weights(relu_bias)

        self.register_buffer("temperature", torch.tensor(1.0))

    def _initialize_weights(self, relu_bias: float) -> None:
        nn.init.zeros_(self.mixing_weights)
        relu_index = self._get_basis_index("relu")
        if relu_index is not None and relu_bias != 0.0:
            self.mixing_weights[:, relu_index].fill_(relu_bias)

    def _get_basis_index(self, name: str) -> int | None:
        try:
            return self.basis_function_names.index(name)
        except ValueError:
            return None

    def set_temperature(self, temperature: float) -> None:
        self.temperature.fill_(temperature)

    def get_mixing_coefficients(self) -> torch.Tensor:
        return F.softmax(self.mixing_weights / self.temperature, dim=-1)

    def compute_entropy(self) -> torch.Tensor:
        alpha = self.get_mixing_coefficients()
        log_alpha = torch.log(alpha + 1e-10)
        entropy = -torch.sum(alpha * log_alpha, dim=-1)
        return entropy.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.get_mixing_coefficients()

        basis_outputs = torch.stack([fn(x) for fn in self.basis_fns], dim=-1)

        output = torch.sum(basis_outputs * alpha, dim=-1)

        return output
