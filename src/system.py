from collections.abc import Callable
from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    spearman_corrcoef,
)

from src.models.actmix_mlp import ActMixMLP

MetricFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]

CLASSIFICATION_METRICS: dict[str, MetricFn] = {
    "accuracy": lambda p, t, n: multiclass_accuracy(
        p, t, num_classes=n, average="micro"
    ),
    "f1": lambda p, t, n: multiclass_f1_score(p, t, num_classes=n, average="macro"),
    "precision": lambda p, t, n: multiclass_precision(
        p, t, num_classes=n, average="macro"
    ),
    "recall": lambda p, t, n: multiclass_recall(p, t, num_classes=n, average="macro"),
}

REGRESSION_METRICS: dict[str, MetricFn] = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmse": lambda p, t: torch.sqrt(mean_squared_error(p, t)),
    "r2": r2_score,
    "pearson": pearson_corrcoef,
    "spearman": spearman_corrcoef,
}


class ActMixSystem(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        task: Literal["regression", "classification"] = "regression",
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        temperature_initial: float = 1.0,
        temperature_final: float = 0.01,
        temperature_anneal_epochs: int = 50,
        entropy_lambda_max: float = 0.1,
        entropy_warmup_epochs: int = 10,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.task = task
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.temperature_initial = temperature_initial
        self.temperature_final = temperature_final
        self.temperature_anneal_epochs = temperature_anneal_epochs

        self.entropy_lambda_max = entropy_lambda_max
        self.entropy_warmup_epochs = entropy_warmup_epochs

        self.is_actmix_model = isinstance(model, ActMixMLP)

        self._preds: dict[str, list[torch.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self._targets: dict[str, list[torch.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return F.cross_entropy(y_hat, y)
        return F.mse_loss(y_hat, y)

    def _get_current_temperature(self) -> float:
        if self.current_epoch >= self.temperature_anneal_epochs:
            return self.temperature_final

        progress = self.current_epoch / self.temperature_anneal_epochs
        return self.temperature_initial - progress * (
            self.temperature_initial - self.temperature_final
        )

    def _get_current_entropy_lambda(self) -> float:
        if self.current_epoch >= self.entropy_warmup_epochs:
            return self.entropy_lambda_max

        progress = self.current_epoch / self.entropy_warmup_epochs
        return progress * self.entropy_lambda_max

    def on_train_epoch_start(self) -> None:
        if self.is_actmix_model:
            temperature = self._get_current_temperature()
            self.model.set_temperature(temperature)
            self.log("temperature", temperature, prog_bar=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.task == "classification":
            metric_preds = torch.argmax(preds, dim=1)
            return {
                name: fn(metric_preds, targets, self.num_classes)
                for name, fn in CLASSIFICATION_METRICS.items()
            }
        return {name: fn(preds, targets) for name, fn in REGRESSION_METRICS.items()}

    def _log_epoch_metrics(self, stage: str) -> None:
        preds_list, targets_list = self._preds[stage], self._targets[stage]

        assert len(preds_list) == len(targets_list)

        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        metrics = self._compute_metrics(all_preds, all_targets)
        primary_metric_name = next(iter(metrics.keys()))
        for name, value in metrics.items():
            is_primary = name == primary_metric_name
            self.log(f"{stage}_{name}", value, prog_bar=is_primary)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        if self.task == "regression":
            y = y.view_as(y_hat)

        task_loss = self._compute_loss(y_hat, y)

        total_loss = task_loss
        entropy_term = torch.tensor(0.0, device=self.device)

        if self.is_actmix_model:
            entropy = self.model.compute_total_entropy()
            entropy_lambda = self._get_current_entropy_lambda()
            entropy_term = entropy_lambda * entropy
            total_loss = task_loss + entropy_term

            self.log("train_entropy", entropy, prog_bar=False)
            self.log("train_entropy_lambda", entropy_lambda, prog_bar=False)

        self.log("train_loss", task_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=False)

        self._preds["train"].append(y_hat.detach().cpu())
        self._targets["train"].append(y.detach().cpu())

        return total_loss

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")
        self._preds["train"].clear()
        self._targets["train"].clear()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)

        if self.task == "regression":
            y = y.view_as(y_hat)

        loss = self._compute_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        self._preds["val"].append(y_hat.detach().cpu())
        self._targets["val"].append(y.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")
        self._preds["val"].clear()
        self._targets["val"].clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)

        if self.task == "regression":
            y = y.view_as(y_hat)

        loss = self._compute_loss(y_hat, y)
        self.log("test_loss", loss)

        self._preds["test"].append(y_hat.detach().cpu())
        self._targets["test"].append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")
        self._preds["test"].clear()
        self._targets["test"].clear()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_mixing_coefficients(self) -> list[torch.Tensor] | None:
        if self.is_actmix_model:
            return self.model.get_all_mixing_coefficients()
        return None
