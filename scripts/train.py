import json
from pathlib import Path
from typing import Any, Literal

import hydra
import lightning as L
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.config import ExperimentConfig
from src.datamodules import DATAMODULE_REGISTRY
from src.models import ActMixMLP, StaticMLP
from src.trainer import TabularTrainer


def create_datamodule(cfg: DictConfig) -> L.LightningDataModule:
    datamodule_class = DATAMODULE_REGISTRY[cfg.dataset.name]
    return datamodule_class(
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
        seed=cfg.seed,
    )


def create_model(cfg: DictConfig, input_dim: int, output_dim: int) -> nn.Module:
    model_name = cfg.model.name
    hidden_dims = list(cfg.model.hidden_dims)
    dropout_rate = cfg.model.dropout_rate

    if model_name == "mlp_actmix":
        return ActMixMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            basis_functions=list(cfg.model.basis_functions),
            dropout_rate=dropout_rate,
            relu_bias=cfg.model.relu_bias,
            omega_0=cfg.model.get("omega_0", 1.0),
        )

    activation = cfg.model.get("activation", "relu")
    return StaticMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout_rate=dropout_rate,
    )


def create_tabular_trainer(
    cfg: DictConfig,
    model: nn.Module,
    task: Literal["regression", "classification"],
) -> TabularTrainer:
    is_actmix = cfg.model.name == "mlp_actmix"

    return TabularTrainer(
        model=model,
        task=task,
        num_classes=cfg.dataset.output_dim,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        temperature_initial=cfg.training.temperature.initial if is_actmix else 1.0,
        temperature_final=cfg.training.temperature.final if is_actmix else 1.0,
        temperature_anneal_epochs=cfg.training.temperature.anneal_epochs
        if is_actmix
        else 1,
        entropy_lambda_max=cfg.training.entropy_regularization.lambda_max
        if is_actmix
        else 0.0,
        entropy_warmup_epochs=cfg.training.entropy_regularization.warmup_epochs
        if is_actmix
        else 0,
    )


def create_callbacks(cfg: DictConfig) -> list[L.Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.training.early_stopping.monitor,
        mode=cfg.training.early_stopping.mode,
        save_top_k=1,
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.early_stopping.monitor,
        patience=cfg.training.early_stopping.patience,
        mode=cfg.training.early_stopping.mode,
    )

    return [
        checkpoint_callback,
        early_stopping_callback,
        RichModelSummary(max_depth=2),
        RichProgressBar(leave=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def create_logger(cfg: DictConfig) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name="",
        version="",
    )


def create_lightning_trainer(
    cfg: DictConfig,
    callbacks: list[L.Callback],
    logger: TensorBoardLogger,
) -> L.Trainer:
    return L.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=True,
        accelerator="auto",
        devices="auto",
    )


def save_experiment_artifacts(cfg: DictConfig, test_results: list[Any]) -> None:
    experiment_dir = Path(cfg.paths.experiment_dir)

    results_payload = {
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "metrics": test_results[0] if test_results else {},
    }

    results_file = experiment_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    config_file = experiment_dir / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    datamodule = create_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    input_dim = datamodule.input_dim
    output_dim = datamodule.output_dim
    task = cfg.dataset.task

    model = create_model(cfg, input_dim, output_dim)
    tabular_trainer = create_tabular_trainer(cfg, model, task)

    callbacks = create_callbacks(cfg)
    logger = create_logger(cfg)
    lightning_trainer = create_lightning_trainer(cfg, callbacks, logger)

    lightning_trainer.fit(tabular_trainer, datamodule=datamodule)
    test_results = lightning_trainer.test(
        tabular_trainer, datamodule=datamodule, ckpt_path="best"
    )

    save_experiment_artifacts(cfg, test_results)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)
    main()
