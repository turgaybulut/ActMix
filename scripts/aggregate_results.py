import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_architecture_info(folder_name: str) -> dict[str, Any]:
    parts = folder_name.split("-")

    if folder_name == "xgboost":
        return {
            "num_layers": None,
            "hidden_dim": None,
            "regularization": None,
        }

    arch_part = parts[0]
    num_layers = int(arch_part.split("x")[0])
    hidden_dim = int(arch_part.split("x")[1])

    reg_str = parts[1] if len(parts) > 1 else None
    if reg_str == "full":
        regularization = True
    elif reg_str == "notemp":
        regularization = False
    else:
        regularization = None

    return {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "regularization": regularization,
    }


def load_cv_results(cv_results_path: Path) -> dict[str, Any] | None:
    if not cv_results_path.exists():
        return None

    with open(cv_results_path) as f:
        return json.load(f)


def determine_task_type(dataset_name: str) -> str:
    classification_datasets = {"adult_census", "magic_gamma"}
    regression_datasets = {"power_plant"}

    if dataset_name in classification_datasets:
        return "classification"
    elif dataset_name in regression_datasets:
        return "regression"
    return "unknown"


def extract_aggregated_metrics(
    aggregated: dict[str, dict[str, float]],
) -> dict[str, float]:
    metrics = {}

    for metric_name, metric_data in aggregated.items():
        clean_name = metric_name.replace("test_", "")
        metrics[f"{clean_name}_mean"] = metric_data["mean"]
        metrics[f"{clean_name}_std"] = metric_data["std"]

    return metrics


def extract_fold_metrics(
    fold_results: list[dict[str, float]],
) -> list[dict[str, float]]:
    cleaned_folds = []

    for fold_data in fold_results:
        cleaned = {}
        for key, value in fold_data.items():
            clean_key = key.replace("test_", "")
            cleaned[clean_key] = value
        cleaned_folds.append(cleaned)

    return cleaned_folds


def collect_all_results(
    models_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregated_results = []
    fold_results = []

    for arch_folder in sorted(models_dir.iterdir()):
        if not arch_folder.is_dir():
            continue

        arch_info = parse_architecture_info(arch_folder.name)
        experiments_dir = arch_folder / "experiments"

        if not experiments_dir.exists():
            continue

        for experiment_folder in sorted(experiments_dir.iterdir()):
            if not experiment_folder.is_dir():
                continue

            cv_results_path = experiment_folder / "cv_results.json"

            if arch_folder.name == "xgboost":
                for sub_experiment in sorted(experiment_folder.iterdir()):
                    if not sub_experiment.is_dir():
                        continue
                    cv_results_path = sub_experiment / "cv_results.json"
                    cv_data = load_cv_results(cv_results_path)

                    if cv_data is None:
                        continue

                    base_info = {
                        "architecture": "xgboost",
                        "num_layers": None,
                        "hidden_dim": None,
                        "regularization": None,
                        "model": cv_data["model"],
                        "dataset": cv_data["dataset"],
                        "task_type": determine_task_type(cv_data["dataset"]),
                        "seed": cv_data["seed"],
                        "n_folds": cv_data["n_folds"],
                    }

                    aggregated_row = {
                        **base_info,
                        **extract_aggregated_metrics(cv_data["aggregated"]),
                    }
                    aggregated_results.append(aggregated_row)

                    fold_metrics = extract_fold_metrics(cv_data["fold_results"])
                    for fold_idx, fold_data in enumerate(fold_metrics):
                        fold_row = {
                            **base_info,
                            "fold": fold_idx,
                            **fold_data,
                        }
                        fold_results.append(fold_row)
            else:
                cv_data = load_cv_results(cv_results_path)

                if cv_data is None:
                    continue

                arch_str = f"{arch_info['num_layers']}x{arch_info['hidden_dim']}"
                base_info = {
                    "architecture": arch_str,
                    "num_layers": arch_info["num_layers"],
                    "hidden_dim": arch_info["hidden_dim"],
                    "regularization": arch_info["regularization"],
                    "model": cv_data["model"],
                    "dataset": cv_data["dataset"],
                    "task_type": determine_task_type(cv_data["dataset"]),
                    "seed": cv_data["seed"],
                    "n_folds": cv_data["n_folds"],
                }

                aggregated_row = {
                    **base_info,
                    **extract_aggregated_metrics(cv_data["aggregated"]),
                }
                aggregated_results.append(aggregated_row)

                fold_metrics = extract_fold_metrics(cv_data["fold_results"])
                for fold_idx, fold_data in enumerate(fold_metrics):
                    fold_row = {
                        **base_info,
                        "fold": fold_idx,
                        **fold_data,
                    }
                    fold_results.append(fold_row)

    return aggregated_results, fold_results


def create_summary_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)

    column_order = [
        "architecture",
        "num_layers",
        "hidden_dim",
        "regularization",
        "model",
        "dataset",
        "task_type",
        "seed",
        "n_folds",
    ]

    metric_columns = [col for col in df.columns if col not in column_order]
    metric_columns = sorted(metric_columns)

    return df[column_order + metric_columns]


def create_fold_dataframe(fold_results: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(fold_results)

    column_order = [
        "architecture",
        "num_layers",
        "hidden_dim",
        "regularization",
        "model",
        "dataset",
        "task_type",
        "seed",
        "n_folds",
        "fold",
    ]

    metric_columns = [col for col in df.columns if col not in column_order]
    metric_columns = sorted(metric_columns)

    return df[column_order + metric_columns]


def format_activation_name(model_name: str) -> str:
    mapping = {
        "mlp_actmix": "ActMix",
        "mlp_relu": "ReLU",
        "mlp_gelu": "GELU",
        "mlp_prelu": "PReLU",
        "xgboost": "XGBoost",
    }
    return mapping.get(model_name, model_name)


def save_results(models_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_results, fold_results = collect_all_results(models_dir)

    if not aggregated_results:
        print("No results found.")
        return

    full_df = create_summary_dataframe(aggregated_results)
    full_csv_path = output_dir / "all_experiments_results.csv"
    full_df.to_csv(full_csv_path, index=False)
    print(f"Saved full results to: {full_csv_path}")

    classification_df = full_df[full_df["task_type"] == "classification"].copy()
    regression_df = full_df[full_df["task_type"] == "regression"].copy()

    if not classification_df.empty:
        classification_cols = [
            "architecture",
            "num_layers",
            "hidden_dim",
            "regularization",
            "model",
            "dataset",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
        ]
        existing_cols = [
            c for c in classification_cols if c in classification_df.columns
        ]
        classification_clean = classification_df[existing_cols]
        classification_csv_path = output_dir / "classification_results.csv"
        classification_clean.to_csv(classification_csv_path, index=False)
        print(f"Saved classification results to: {classification_csv_path}")

    if not regression_df.empty:
        regression_cols = [
            "architecture",
            "num_layers",
            "hidden_dim",
            "regularization",
            "model",
            "dataset",
            "mse_mean",
            "mse_std",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "r2_mean",
            "r2_std",
            "pearson_mean",
            "pearson_std",
            "spearman_mean",
            "spearman_std",
        ]
        existing_cols = [c for c in regression_cols if c in regression_df.columns]
        regression_clean = regression_df[existing_cols]
        regression_csv_path = output_dir / "regression_results.csv"
        regression_clean.to_csv(regression_csv_path, index=False)
        print(f"Saved regression results to: {regression_csv_path}")

    if fold_results:
        fold_df = create_fold_dataframe(fold_results)

        fold_csv_path = output_dir / "all_fold_results.csv"
        fold_df.to_csv(fold_csv_path, index=False)
        print(f"Saved all fold results to: {fold_csv_path}")

        classification_fold_df = fold_df[
            fold_df["task_type"] == "classification"
        ].copy()
        regression_fold_df = fold_df[fold_df["task_type"] == "regression"].copy()

        if not classification_fold_df.empty:
            classification_fold_cols = [
                "architecture",
                "num_layers",
                "hidden_dim",
                "regularization",
                "model",
                "dataset",
                "fold",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "loss",
            ]
            existing_cols = [
                c
                for c in classification_fold_cols
                if c in classification_fold_df.columns
            ]
            classification_fold_clean = classification_fold_df[existing_cols]
            cls_fold_csv_path = output_dir / "classification_fold_results.csv"
            classification_fold_clean.to_csv(cls_fold_csv_path, index=False)
            print(f"Saved classification fold results to: {cls_fold_csv_path}")

        if not regression_fold_df.empty:
            regression_fold_cols = [
                "architecture",
                "num_layers",
                "hidden_dim",
                "regularization",
                "model",
                "dataset",
                "fold",
                "mse",
                "mae",
                "rmse",
                "r2",
                "pearson",
                "spearman",
                "loss",
            ]
            existing_cols = [
                c for c in regression_fold_cols if c in regression_fold_df.columns
            ]
            regression_fold_clean = regression_fold_df[existing_cols]
            regression_fold_csv_path = output_dir / "regression_fold_results.csv"
            regression_fold_clean.to_csv(regression_fold_csv_path, index=False)
            print(f"Saved regression fold results to: {regression_fold_csv_path}")

    print_summary_tables(full_df)


def print_summary_tables(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    for task_type in ["classification", "regression"]:
        task_df = df[df["task_type"] == task_type]

        if task_df.empty:
            continue

        print(f"\n{task_type.upper()} RESULTS")
        print("-" * 60)

        for dataset in task_df["dataset"].unique():
            dataset_df = task_df[task_df["dataset"] == dataset]

            print(f"\nDataset: {dataset}")
            print("-" * 40)

            if task_type == "classification":
                primary_metric = "accuracy_mean"
                metric_name = "Accuracy"
            else:
                primary_metric = "r2_mean"
                metric_name = "R²"

            if primary_metric not in dataset_df.columns:
                continue

            sorted_df = dataset_df.sort_values(primary_metric, ascending=False)

            for _, row in sorted_df.iterrows():
                reg = row.get("regularization")
                if pd.notna(reg):
                    reg_str = "reg" if reg else "noreg"
                    config = f"{row['architecture']}-{reg_str}"
                else:
                    config = row["architecture"]
                activation = format_activation_name(row["model"])
                mean_val = row[primary_metric]
                std_col = primary_metric.replace("_mean", "_std")
                std_val = row.get(std_col, 0)

                print(
                    f"  {config:15} {activation:10} "
                    f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}"
                )


def main() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    output_dir = project_root / "results"

    save_results(models_dir, output_dir)


if __name__ == "__main__":
    main()
