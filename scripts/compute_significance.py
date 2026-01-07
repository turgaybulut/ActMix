import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main():
    project_root = Path(__file__).parent.parent
    results_path = project_root / "results" / "all_fold_results.csv"
    output_path = project_root / "results" / "significance.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        sys.exit(1)
    df = pd.read_csv(results_path)
    datasets = df["dataset"].unique()
    results = {}
    models = ["mlp_actmix", "mlp_gelu", "mlp_relu", "mlp_prelu"]
    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset].copy()
        if dataset == "power_plant":
            metric = "r2"
            maximize = True
        else:
            metric = "accuracy"
            maximize = True
        best_configs = {}
        for model in models:
            if model == "mlp_actmix":
                model_df = ds_df[(ds_df["model"] == model) & (ds_df["regularization"])]
            else:
                model_df = ds_df[ds_df["model"] == model]
            if model_df.empty:
                continue
            agg = model_df.groupby("architecture")[metric].mean()
            if maximize:
                best_arch = agg.idxmax()
            else:
                best_arch = agg.idxmin()
            best_configs[model] = best_arch
        filtered_df = pd.DataFrame()
        for model, arch in best_configs.items():
            if model == "mlp_actmix":
                subset = ds_df[
                    (ds_df["model"] == model)
                    & (ds_df["architecture"] == arch)
                    & (ds_df["regularization"])
                ]
            else:
                subset = ds_df[
                    (ds_df["model"] == model) & (ds_df["architecture"] == arch)
                ]
            filtered_df = pd.concat([filtered_df, subset])
        filtered_df = filtered_df.groupby(["fold", "model"], as_index=False)[
            metric
        ].mean()
        pivot_df = filtered_df.pivot(index="fold", columns="model", values=metric)
        neural_models = [c for c in pivot_df.columns if c in models]
        neural_df = pivot_df[neural_models].dropna()
        if neural_df.empty:
            continue
        try:
            stat, p_value = stats.friedmanchisquare(
                *[neural_df[m] for m in neural_models]
            )
            friedman_res = {"statistic": stat, "p_value": p_value}
        except ValueError as e:
            friedman_res = {"error": str(e)}
        pairwise_res = {}
        target_model = "mlp_actmix"
        if target_model in neural_models:
            others = [m for m in neural_models if m != target_model]
            p_values = []
            comparisons = []
            for other in others:
                try:
                    stat, p = stats.wilcoxon(neural_df[target_model], neural_df[other])
                    p_values.append(p)
                    comparisons.append(other)
                except Exception:
                    pass
            if p_values:
                sorted_indices = np.argsort(p_values)
                sorted_p = np.array(p_values)[sorted_indices]
                sorted_comp = np.array(comparisons)[sorted_indices]
                m = len(p_values)
                corrected_p = []
                for i, p in enumerate(sorted_p):
                    if i == 0:
                        adj = min(1.0, p * m)
                    else:
                        adj = min(1.0, max(corrected_p[-1], p * (m - i)))
                    corrected_p.append(adj)
                final_p_map = {}
                for k, comp in enumerate(sorted_comp):
                    final_p_map[comp] = corrected_p[k]
                pairwise_res = final_p_map
        results[dataset] = {
            "metric": metric,
            "friedman": friedman_res,
            "wilcoxon_holm_adjusted": pairwise_res,
        }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Significance tests complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
