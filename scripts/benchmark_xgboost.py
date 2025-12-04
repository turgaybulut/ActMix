import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
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

from src.datamodules.uci_classification import (
    AdultCensusDataModule,
    MagicGammaDataModule,
)
from src.datamodules.uci_regression import PowerPlantDataModule


def get_numpy_data(datamodule) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datamodule.prepare_data()
    datamodule.setup()

    X_train = datamodule.train_dataset.features.numpy()
    y_train = datamodule.train_dataset.targets.numpy()

    X_test = datamodule.test_dataset.features.numpy()
    y_test = datamodule.test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test


def run_benchmark(
    name: str, datamodule_cls: Any, task: str, params: dict[str, Any] = None
) -> dict[str, float]:
    print(f"Running benchmark for {name}...")

    datamodule = datamodule_cls()
    X_train, y_train, X_test, y_test = get_numpy_data(datamodule)

    if task == "classification":
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            **params if params else {},
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            **params if params else {},
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task == "classification":
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")

        result = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
    else:
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        pearson, _ = pearsonr(y_test.flatten(), preds.flatten())
        spearman, _ = spearmanr(y_test.flatten(), preds.flatten())

        result = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "pearson": float(pearson),
            "spearman": float(spearman),
        }

    return result


def main():
    results = {}

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

    table = Table(title="XGBoost Benchmark Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Task", style="magenta")
    table.add_column("Metric", style="green")
    table.add_column("Score", style="yellow")

    for name, dm_cls, task, params in benchmarks:
        try:
            metrics = run_benchmark(name, dm_cls, task, params)
            results[name] = metrics

            if task == "classification":
                table.add_row(name, task, "Accuracy", f"{metrics['accuracy']:.4f}")
                table.add_row("", "", "F1", f"{metrics['f1']:.4f}")
                table.add_row("", "", "Precision", f"{metrics['precision']:.4f}")
                table.add_row("", "", "Recall", f"{metrics['recall']:.4f}")
            else:
                table.add_row(name, task, "RMSE", f"{metrics['rmse']:.4f}")
                table.add_row("", "", "MSE", f"{metrics['mse']:.4f}")
                table.add_row("", "", "MAE", f"{metrics['mae']:.4f}")
                table.add_row("", "", "R2", f"{metrics['r2']:.4f}")
                table.add_row("", "", "Pearson", f"{metrics['pearson']:.4f}")
                table.add_row("", "", "Spearman", f"{metrics['spearman']:.4f}")

        except Exception as e:
            print(f"Failed to run benchmark for {name}: {e}")
            table.add_row(name, task, "Error", str(e))

    print(table)

    output_dir = Path("experiments/xgboost_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
