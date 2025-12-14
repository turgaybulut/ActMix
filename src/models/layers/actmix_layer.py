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
        "sigmoid": torch.sigmoid,
        "silu": F.silu,
    }

    def __init__(
        self,
        num_features: int,
        basis_functions: list[str] | None = None,
        relu_bias: float = 1.5,
        omega_0: float = 1.0,
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
        self.omega_0 = omega_0

        self._validate_basis_functions()

        self.mixing_logits = nn.Parameter(torch.zeros(num_features, self.num_basis))
        self._initialize_mixing_logits(relu_bias)

        self.register_buffer("temperature", torch.tensor(1.0))

    def _validate_basis_functions(self) -> None:
        for name in self.basis_function_names:
            assert name in self.BASIS_FUNCTIONS, f"Unknown basis function: {name}"

    def _initialize_mixing_logits(self, relu_bias: float) -> None:
        with torch.no_grad():
            self.mixing_logits.zero_()
            relu_idx = self._get_basis_index("relu")
            if relu_idx is not None and relu_bias != 0.0:
                self.mixing_logits[:, relu_idx].fill_(relu_bias)

    def _get_basis_index(self, name: str) -> int | None:
        try:
            return self.basis_function_names.index(name)
        except ValueError:
            return None

    def set_temperature(self, temperature: float) -> None:
        self.temperature.fill_(temperature)

    def get_mixing_coefficients(self) -> torch.Tensor:
        return F.softmax(self.mixing_logits / self.temperature, dim=-1)

    def _get_log_mixing_coefficients(self) -> torch.Tensor:
        return F.log_softmax(self.mixing_logits / self.temperature, dim=-1)

    def compute_entropy(self) -> torch.Tensor:
        log_alpha = self._get_log_mixing_coefficients()
        alpha = self.get_mixing_coefficients()
        entropy = -torch.sum(alpha * log_alpha, dim=-1)
        return entropy.mean()

    def _apply_basis_function(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "sin":
            return torch.sin(self.omega_0 * x)
        return self.BASIS_FUNCTIONS[name](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.get_mixing_coefficients()

        basis_outputs = torch.stack(
            [self._apply_basis_function(x, name) for name in self.basis_function_names],
            dim=-1,
        )

        return torch.sum(basis_outputs * alpha, dim=-1)
