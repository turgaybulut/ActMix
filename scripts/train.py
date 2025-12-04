import json
from pathlib import Path
from typing import Any, Literal

import hydra
import lightning as L
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from src.datamodules.uci_classification import (
    AdultCensusDataModule,
    MagicGammaDataModule,
)
from src.datamodules.uci_regression import PowerPlantDataModule
from src.models.actmix_mlp import ActMixMLP
from src.models.baselines import GeLUMLP, PReLUMLP, ReLUMLP
from src.system import ActMixSystem

DATAMODULE_REGISTRY = {
    "power_plant": PowerPlantDataModule,
    "adult_census": AdultCensusDataModule,
    "magic_gamma": MagicGammaDataModule,
}

MODEL_REGISTRY = {
    "actmix": ActMixMLP,
    "mlp_relu": ReLUMLP,
    "mlp_gelu": GeLUMLP,
    "mlp_prelu": PReLUMLP,
}


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
    model_class = MODEL_REGISTRY[model_name]

    common_kwargs = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": list(cfg.model.hidden_dims),
        "dropout_rate": cfg.model.dropout_rate,
    }

    if model_name == "actmix":
        common_kwargs["basis_functions"] = list(cfg.model.basis_functions)
        common_kwargs["relu_bias"] = cfg.model.relu_bias

    return model_class(**common_kwargs)


def create_system(
    cfg: DictConfig,
    model: nn.Module,
    task: Literal["regression", "classification"],
) -> ActMixSystem:
    is_actmix = cfg.model.name == "actmix"

    return ActMixSystem(
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

    rich_model_summary = RichModelSummary(max_depth=2)

    rich_progress_bar = RichProgressBar(leave=True)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [
        checkpoint_callback,
        early_stopping_callback,
        rich_model_summary,
        rich_progress_bar,
        lr_monitor,
    ]


def create_logger(cfg: DictConfig) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name="",
        version="",
    )


def create_trainer(
    cfg: DictConfig, callbacks: list[L.Callback], logger: TensorBoardLogger
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


def save_test_results(test_results: list[Any], cfg: DictConfig) -> None:
    results_file = Path(cfg.paths.experiment_dir) / "test_results.json"

    results_payload = {
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "metrics": test_results[0] if test_results else {},
    }

    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    datamodule = create_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    input_dim = datamodule.input_dim
    output_dim = datamodule.output_dim
    task = cfg.dataset.task

    model = create_model(cfg, input_dim, output_dim)
    system = create_system(cfg, model, task)

    callbacks = create_callbacks(cfg)
    logger = create_logger(cfg)
    trainer = create_trainer(cfg, callbacks, logger)

    trainer.fit(system, datamodule=datamodule)
    test_results = trainer.test(system, datamodule=datamodule, ckpt_path="best")

    save_test_results(test_results, cfg)


if __name__ == "__main__":
    main()
