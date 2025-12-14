from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemperatureConfig:
    initial: float = 1.0
    final: float = 0.01
    anneal_epochs: int = 150


@dataclass
class EntropyRegularizationConfig:
    lambda_max: float = 0.01
    warmup_epochs: int = 50


@dataclass
class EarlyStoppingConfig:
    patience: int = 20
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class TrainingConfig:
    seed: int = 1192
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 0.5
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    entropy_regularization: EntropyRegularizationConfig = field(
        default_factory=EntropyRegularizationConfig
    )
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class DatasetConfig:
    name: str = ""
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.10
    test_split: float = 0.10
    seed: int = 1192
    task: str = ""
    input_dim: int = 0
    output_dim: int = 0


@dataclass
class ActMixModelConfig:
    name: str = "actmix"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    basis_functions: list[str] = field(
        default_factory=lambda: ["sin", "tanh", "relu", "identity"]
    )
    dropout_rate: float = 0.1
    relu_bias: float = 1.5
    omega_0: float = 1.0


@dataclass
class StaticMLPModelConfig:
    name: str = "mlp_relu"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    dropout_rate: float = 0.1


@dataclass
class PathsConfig:
    root_dir: str = "."
    experiment_dir: str = ""
    log_dir: str = ""
    checkpoint_dir: str = ""


@dataclass
class ExperimentConfig:
    seed: int = 1192
    experiment_name: str = ""
    cv_folds: int = 10
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: Any = field(default_factory=ActMixModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
