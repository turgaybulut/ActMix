from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import compute_metrics, get_primary_metric
from src.models.actmix_mlp import ActMixMLP
from src.schedulers import EntropyLambdaScheduler, TemperatureScheduler


class TabularTrainer(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        task: Literal["regression", "classification"] = "regression",
        num_classes: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        eta_min_factor: float = 0.01,
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
        self.eta_min_factor = eta_min_factor

        self.temperature_scheduler = TemperatureScheduler(
            initial=temperature_initial,
            final=temperature_final,
            anneal_epochs=temperature_anneal_epochs,
        )

        self.entropy_scheduler = EntropyLambdaScheduler(
            lambda_max=entropy_lambda_max,
            warmup_epochs=entropy_warmup_epochs,
        )

        self.is_actmix_model = isinstance(model, ActMixMLP)
        self.primary_metric = get_primary_metric(task)

        self._epoch_predictions: dict[str, list[torch.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self._epoch_targets: dict[str, list[torch.Tensor]] = {
            "train": [],
            "val": [],
            "test": [],
        }

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.task == "classification":
            return F.cross_entropy(predictions, targets)
        return F.mse_loss(predictions, targets)

    def on_train_epoch_start(self) -> None:
        if self.is_actmix_model:
            temperature = self.temperature_scheduler.get_temperature(self.current_epoch)
            self.model.set_temperature(temperature)
            self.log("temperature", temperature, prog_bar=True, sync_dist=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _accumulate_predictions(
        self,
        stage: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._epoch_predictions[stage].append(predictions.detach())
        self._epoch_targets[stage].append(targets.detach())

    def _gather_and_compute_metrics(self, stage: str) -> None:
        local_predictions = torch.cat(self._epoch_predictions[stage], dim=0)
        local_targets = torch.cat(self._epoch_targets[stage], dim=0)

        if self.trainer.world_size > 1:
            gathered_predictions = self.all_gather(local_predictions)
            gathered_targets = self.all_gather(local_targets)

            gathered_predictions = gathered_predictions.view(
                -1, *local_predictions.shape[1:]
            )
            gathered_targets = gathered_targets.view(-1, *local_targets.shape[1:])
        else:
            gathered_predictions = local_predictions
            gathered_targets = local_targets

        if self.trainer.is_global_zero:
            metrics = compute_metrics(
                self.task,
                gathered_predictions.cpu(),
                gathered_targets.cpu(),
                self.num_classes,
            )

            for name, value in metrics.items():
                is_primary = name == self.primary_metric
                self.log(
                    f"{stage}_{name}",
                    value,
                    prog_bar=is_primary,
                    rank_zero_only=True,
                )

        self._epoch_predictions[stage].clear()
        self._epoch_targets[stage].clear()

    def _prepare_targets(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.task == "regression":
            return targets.view_as(predictions)
        return targets

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        features, targets = batch
        predictions = self(features)
        targets = self._prepare_targets(predictions, targets)
        batch_size = features.shape[0]

        task_loss = self._compute_loss(predictions, targets)
        total_loss = task_loss

        if self.is_actmix_model:
            entropy = self.model.compute_total_entropy()
            entropy_lambda = self.entropy_scheduler.get_lambda(self.current_epoch)
            entropy_term = entropy_lambda * entropy
            total_loss = task_loss + entropy_term

            self.log("train_entropy", entropy, batch_size=batch_size, sync_dist=True)
            self.log(
                "train_entropy_lambda",
                entropy_lambda,
                batch_size=batch_size,
                sync_dist=True,
            )

        self.log(
            "train_loss",
            task_loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log("train_total_loss", total_loss, batch_size=batch_size, sync_dist=True)

        self._accumulate_predictions("train", predictions, targets)

        return total_loss

    def on_train_epoch_end(self) -> None:
        self._gather_and_compute_metrics("train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        features, targets = batch
        predictions = self(features)
        targets = self._prepare_targets(predictions, targets)
        batch_size = features.shape[0]

        loss = self._compute_loss(predictions, targets)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)

        self._accumulate_predictions("val", predictions, targets)

    def on_validation_epoch_end(self) -> None:
        self._gather_and_compute_metrics("val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        features, targets = batch
        predictions = self(features)
        targets = self._prepare_targets(predictions, targets)
        batch_size = features.shape[0]

        loss = self._compute_loss(predictions, targets)
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)

        self._accumulate_predictions("test", predictions, targets)

    def on_test_epoch_end(self) -> None:
        self._gather_and_compute_metrics("test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * self.eta_min_factor,
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
