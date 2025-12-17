import json
from pathlib import Path
from typing import Literal

import hydra
import lightning as L
import numpy as np
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
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import KFold, StratifiedKFold

from src.config import ExperimentConfig
from src.datamodules import DATAMODULE_REGISTRY
from src.datamodules.base import TabularDataset
from src.models import ActMixMLP, StaticMLP
from src.trainer import TabularTrainer


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
        eta_min_factor=cfg.training.scheduler.eta_min_factor,
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


def create_callbacks(cfg: DictConfig, fold: int) -> list[L.Callback]:
    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / f"fold_{fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
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


def create_logger(cfg: DictConfig, fold: int) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name=f"fold_{fold}",
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
        deterministic=cfg.training.deterministic,
        accelerator=cfg.training.distributed.accelerator,
        devices=cfg.training.distributed.devices,
        strategy=cfg.training.distributed.strategy,
        enable_model_summary=False,
    )


def run_fold(
    cfg: DictConfig,
    fold: int,
    train_dataset: TabularDataset,
    val_dataset: TabularDataset,
    input_dim: int,
    output_dim: int,
    task: str,
) -> dict[str, float]:
    console = Console()
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]Fold {fold + 1}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    model = create_model(cfg, input_dim, output_dim)
    tabular_trainer = create_tabular_trainer(cfg, model, task)

    callbacks = create_callbacks(cfg, fold)
    logger = create_logger(cfg, fold)
    lightning_trainer = create_lightning_trainer(cfg, callbacks, logger)

    train_loader = train_dataset_to_loader(train_dataset, cfg, shuffle=True)
    val_loader = train_dataset_to_loader(val_dataset, cfg, shuffle=False)

    lightning_trainer.fit(
        tabular_trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    test_results = lightning_trainer.test(
        tabular_trainer,
        dataloaders=val_loader,
        ckpt_path="best",
    )

    return test_results[0] if test_results else {}


def train_dataset_to_loader(
    dataset: TabularDataset,
    cfg: DictConfig,
    shuffle: bool,
) -> L.LightningDataModule:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.num_workers > 0,
    )


def aggregate_results(
    fold_results: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    all_metrics: dict[str, list[float]] = {}

    for fold_result in fold_results:
        for metric_name, value in fold_result.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    aggregated = {}
    for metric_name, values in all_metrics.items():
        aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    return aggregated


def print_results_table(aggregated: dict[str, dict[str, float]], n_folds: int) -> None:
    console = Console()

    table = Table(title=f"Cross-Validation Results ({n_folds} Folds)")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean ± Std", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="yellow")

    for metric_name, stats in aggregated.items():
        table.add_row(
            metric_name,
            f"{stats['mean']:.4f} ± {stats['std']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
        )

    console.print(table)


def save_cv_results(
    cfg: DictConfig,
    fold_results: list[dict[str, float]],
    aggregated: dict[str, dict[str, float]],
    n_folds: int,
) -> None:
    experiment_dir = Path(cfg.paths.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "aggregated": aggregated,
    }

    results_file = experiment_dir / "cv_results.json"
    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    config_file = experiment_dir / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    n_folds = cfg.cv_folds

    L.seed_everything(cfg.seed, workers=True)

    datamodule_class = DATAMODULE_REGISTRY[cfg.dataset.name]
    datamodule = datamodule_class(
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        val_split=0.0,
        test_split=0.0,
        seed=cfg.seed,
        pin_memory=cfg.dataset.pin_memory,
    )
    datamodule.prepare_data()
    datamodule.setup()

    X = datamodule.train_dataset.features.numpy()
    y = datamodule.train_dataset.targets.numpy()

    input_dim = datamodule.input_dim
    output_dim = datamodule.output_dim
    task = cfg.dataset.task
    target_dtype = datamodule.target_dtype

    if task == "classification":
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
        split_iterator = kfold.split(X, y)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
        split_iterator = kfold.split(X)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TabularDataset(X_train, y_train, target_dtype)
        val_dataset = TabularDataset(X_val, y_val, target_dtype)

        fold_result = run_fold(
            cfg=cfg,
            fold=fold,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            task=task,
        )
        fold_results.append(fold_result)

    aggregated = aggregate_results(fold_results)
    print_results_table(aggregated, n_folds)
    save_cv_results(cfg, fold_results, aggregated, n_folds)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)
    main()
