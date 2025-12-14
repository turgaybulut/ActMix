from collections.abc import Callable
from typing import Literal

import torch
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

ClassificationMetricFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
RegressionMetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

CLASSIFICATION_METRICS: dict[str, ClassificationMetricFn] = {
    "accuracy": lambda p, t, n: multiclass_accuracy(
        p, t, num_classes=n, average="micro"
    ),
    "f1": lambda p, t, n: multiclass_f1_score(p, t, num_classes=n, average="macro"),
    "precision": lambda p, t, n: multiclass_precision(
        p, t, num_classes=n, average="macro"
    ),
    "recall": lambda p, t, n: multiclass_recall(p, t, num_classes=n, average="macro"),
}

REGRESSION_METRICS: dict[str, RegressionMetricFn] = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmse": lambda p, t: torch.sqrt(mean_squared_error(p, t)),
    "r2": r2_score,
    "pearson": pearson_corrcoef,
    "spearman": spearman_corrcoef,
}


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    pred_classes = torch.argmax(predictions, dim=1)
    return {
        name: fn(pred_classes, targets, num_classes)
        for name, fn in CLASSIFICATION_METRICS.items()
    }


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {name: fn(predictions, targets) for name, fn in REGRESSION_METRICS.items()}


def compute_metrics(
    task: Literal["classification", "regression"],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 1,
) -> dict[str, torch.Tensor]:
    if task == "classification":
        return compute_classification_metrics(predictions, targets, num_classes)
    return compute_regression_metrics(predictions, targets)


def get_primary_metric(task: Literal["classification", "regression"]) -> str:
    return "accuracy" if task == "classification" else "mse"
