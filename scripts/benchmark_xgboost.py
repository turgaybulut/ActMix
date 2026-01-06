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

SEED = 1192
N_FOLDS = 10


def get_all_data(datamodule_cls: Any) -> tuple[np.ndarray, np.ndarray]:
    datamodule = datamodule_cls(val_split=0.0, test_split=0.0, seed=SEED)
    datamodule.prepare_data()
    datamodule.setup()

    X = datamodule.train_dataset.features.numpy()
    y = datamodule.train_dataset.targets.numpy()

    return X, y


def evaluate_classification(y_test: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds, average="macro")),
        "test_precision": float(precision_score(y_test, preds, average="macro")),
        "test_recall": float(recall_score(y_test, preds, average="macro")),
    }


def evaluate_regression(y_test: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_test, preds)
    return {
        "test_mse": float(mse),
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(np.sqrt(mse)),
        "test_r2": float(r2_score(y_test, preds)),
        "test_pearson": float(pearsonr(y_test.flatten(), preds.flatten())[0]),
        "test_spearman": float(spearmanr(y_test.flatten(), preds.flatten())[0]),
    }


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


def run_cv_benchmark(
    name: str,
    datamodule_cls: Any,
    task: str,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    import xgboost as xgb

    console = Console()
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]Dataset: {name}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")

    X, y = get_all_data(datamodule_cls)
    console.print(f"Data shape: X={X.shape}, y={y.shape}")

    if task == "classification":
        kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = kfold.split(X, y)
    else:
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = kfold.split(X)

    fold_results: list[dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        console.print(f"Running fold {fold_idx + 1}/{N_FOLDS}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if task == "classification":
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
                **(params or {}),
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED,
                n_jobs=-1,
                **(params or {}),
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "classification":
            metrics = evaluate_classification(y_test, preds)
        else:
            metrics = evaluate_regression(y_test, preds)

        fold_results.append(metrics)

    aggregated = aggregate_results(fold_results)
    return fold_results, aggregated


def save_cv_results(
    output_dir: Path,
    dataset_name: str,
    fold_results: list[dict[str, float]],
    aggregated: dict[str, dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "model": "xgboost",
        "dataset": dataset_name,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "fold_results": fold_results,
        "aggregated": aggregated,
    }

    results_file = output_dir / "cv_results.json"
    with open(results_file, "w") as f:
        json.dump(results_payload, f, indent=2)


def print_results_table(
    aggregated: dict[str, dict[str, float]],
    dataset_name: str,
) -> None:
    console = Console()

    table = Table(title=f"XGBoost {N_FOLDS}-Fold CV Results: {dataset_name}")
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


def main():
    benchmarks = [
        ("power_plant", PowerPlantDataModule, "regression", None),
        ("magic_gamma", MagicGammaDataModule, "classification", None),
        (
            "adult_census",
            AdultCensusDataModule,
            "classification",
            {"enable_categorical": True},
        ),
    ]

    base_output_dir = Path("experiments/xgboost_baselines")
    console = Console()

    for dataset_name, dm_cls, task, params in benchmarks:
        try:
            fold_results, aggregated = run_cv_benchmark(
                dataset_name, dm_cls, task, params
            )

            print_results_table(aggregated, dataset_name)

            output_dir = base_output_dir / f"xgboost-{dataset_name}"
            save_cv_results(output_dir, dataset_name, fold_results, aggregated)

            results_path = output_dir / "cv_results.json"
            console.print(f"[green]Results saved to {results_path}[/green]\n")

        except Exception as e:
            console.print(f"[red]Failed for {dataset_name}: {e}[/red]")
            raise


if __name__ == "__main__":
    main()
