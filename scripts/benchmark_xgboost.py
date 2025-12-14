import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from src.datamodules.uci_classification import (
    AdultCensusDataModule,
    MagicGammaDataModule,
)
from src.datamodules.uci_regression import PowerPlantDataModule


def get_all_data(datamodule_cls: Any) -> tuple[np.ndarray, np.ndarray]:
    datamodule = datamodule_cls(val_split=0.0, test_split=0.0, seed=42)
    datamodule.prepare_data()
    datamodule.setup()

    X = datamodule.train_dataset.features.numpy()
    y = datamodule.train_dataset.targets.numpy()

    return X, y


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, task: str
) -> dict[str, float]:
    preds = model.predict(X_test)

    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, average="macro")),
            "precision": float(precision_score(y_test, preds, average="macro")),
            "recall": float(recall_score(y_test, preds, average="macro")),
        }

    mse = mean_squared_error(y_test, preds)
    return {
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, preds)),
        "pearson": float(pearsonr(y_test.flatten(), preds.flatten())[0]),
        "spearman": float(spearmanr(y_test.flatten(), preds.flatten())[0]),
    }


def run_cv_benchmark(
    name: str,
    datamodule_cls: Any,
    task: str,
    params: dict[str, Any] = None,
    n_folds: int = 10,
) -> dict[str, dict[str, float]]:
    import xgboost as xgb  # Lazy import to avoid OpenMP conflicts with Torch

    print(f"Loading data for {name}...")
    X, y = get_all_data(datamodule_cls)
    print(f"Data loaded for {name}. Shape: {X.shape}, {y.shape}")

    if task == "classification":
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kfold.split(X, y)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kfold.split(X)

    fold_metrics: list[dict[str, float]] = []

    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Running fold {i + 1}/{n_folds}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if task == "classification":
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
                **(params if params else {}),
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1,
                **(params if params else {}),
            )

        model.fit(X_train, y_train)
        fold_metrics.append(evaluate_model(model, X_test, y_test, task))

    aggregated = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return aggregated


def main():
    benchmarks = [
        ("Power Plant", PowerPlantDataModule, "regression", {}),
        ("Magic Gamma", MagicGammaDataModule, "classification", {}),
        (
            "Adult Census",
            AdultCensusDataModule,
            "classification",
            {"enable_categorical": True},
        ),
    ]

    results = {}
    console = Console()

    table = Table(title="XGBoost 10-Fold CV Benchmark Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Metric", style="green")
    table.add_column("Mean ± Std", style="yellow")

    for name, dm_cls, task, params in benchmarks:
        try:
            metrics = run_cv_benchmark(name, dm_cls, task, params)
            results[name] = metrics

            if task == "classification":
                row_metrics = ["accuracy", "f1", "precision", "recall"]
                names = ["Accuracy", "F1", "Precision", "Recall"]
            else:
                row_metrics = ["rmse", "mse", "mae", "r2", "pearson", "spearman"]
                names = ["RMSE", "MSE", "MAE", "R2", "Pearson", "Spearman"]

            for i, (key, display_name) in enumerate(zip(row_metrics, names)):
                m = metrics[key]
                val_str = f"{m['mean']:.4f} ± {m['std']:.4f}"
                if i == 0:
                    table.add_row(name, task, display_name, val_str)
                else:
                    table.add_row("", "", display_name, val_str)

        except Exception as e:
            console.print(f"[red]Failed to run benchmark for {name}: {e}[/red]")
            table.add_row(name, task, "Error", str(e))

    console.print(table)

    output_dir = Path("experiments/xgboost_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
