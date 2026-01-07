import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from scripts.analyze_weights import (
        compute_layer_statistics,
        extract_actmix_layers,
        get_entropy_per_neuron,
        get_mixing_coefficients,
        load_model_from_checkpoint,
    )
except ImportError:
    print(
        "Could not import from analyze_weights. Please ensure scripts/analyze_weights.py exists."  # noqa: E501
    )
    sys.exit(1)


def aggregate_statistics(all_fold_stats):
    layer_names = list(all_fold_stats[0].keys())
    aggregated = {}
    for layer in layer_names:
        layer_stats = {
            "mean_entropy": [],
            "max_alphas": [],
            "basis_mass": [],
            "dominance_counts": {},
        }
        first_fold_counts = all_fold_stats[0][layer]["dominance_counts"]
        for key in first_fold_counts:
            layer_stats["dominance_counts"][key] = []
        for fold_data in all_fold_stats:
            stats = fold_data[layer]
            layer_stats["mean_entropy"].append(stats["mean_entropy"])
            layer_stats["max_alphas"].extend(stats["max_alphas"])
            layer_stats["basis_mass"].append(stats["basis_mass"])
            for key, count in stats["dominance_counts"].items():
                layer_stats["dominance_counts"][key].append(count)
        agg_layer = {
            "mean_entropy_mean": np.mean(layer_stats["mean_entropy"]),
            "mean_entropy_std": np.std(layer_stats["mean_entropy"]),
            "max_alphas_all": layer_stats["max_alphas"],
            "basis_mass_mean": np.mean(layer_stats["basis_mass"], axis=0).tolist(),
            "basis_mass_std": np.std(layer_stats["basis_mass"], axis=0).tolist(),
            "dominance_counts_mean": {
                k: np.mean(v) for k, v in layer_stats["dominance_counts"].items()
            },
            "dominance_counts_std": {
                k: np.std(v) for k, v in layer_stats["dominance_counts"].items()
            },
        }
        aggregated[layer] = agg_layer
    return aggregated


def plot_aggregated_topology(aggregated_stats, output_dir):
    layer_names = list(aggregated_stats.keys())
    basis_names = list(aggregated_stats[layer_names[0]]["dominance_counts_mean"].keys())
    mass_mean = {name: [] for name in basis_names}
    for layer in layer_names:
        counts = aggregated_stats[layer]["dominance_counts_mean"]
        for name in basis_names:
            mass_mean[name].append(counts[name])
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(layer_names))
    for name in basis_names:
        values = np.array(mass_mean[name])
        ax.bar(layer_names, values, bottom=bottom, label=name)
        bottom += values
    plt.xlabel("Layer")
    plt.ylabel("Average Dominant Neurons Count")
    plt.title("Layer-wise Topology Evolution (Mean across Folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "layer_topology_evolution.pdf")
    plt.savefig(output_dir / "layer_topology_evolution.png", dpi=150)
    plt.close()


def plot_aggregated_specialization(aggregated_stats, output_dir):
    all_alphas = []
    for layer in aggregated_stats:
        all_alphas.extend(aggregated_stats[layer]["max_alphas_all"])
    plt.figure(figsize=(10, 6))
    plt.hist(
        all_alphas, bins=50, range=(0, 1), edgecolor="black", alpha=0.7, density=True
    )
    plt.axvline(0.25, color="red", linestyle="--", label="Mean-Field (0.25)")
    plt.axvline(1.0, color="green", linestyle="--", label="Specialized (1.0)")
    plt.xlabel("Max Mixing Coefficient")
    plt.ylabel("Density")
    plt.title("Neuron Specialization Spectrum (All Layers, All Folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "specialization_spectrum.pdf")
    plt.savefig(output_dir / "specialization_spectrum.png", dpi=150)
    plt.close()


def plot_aggregated_entropy(aggregated_stats, output_dir):
    layer_names = list(aggregated_stats.keys())
    means = [aggregated_stats[layer]["mean_entropy_mean"] for layer in layer_names]
    stds = [aggregated_stats[layer]["mean_entropy_std"] for layer in layer_names]
    plt.figure(figsize=(10, 6))
    plt.errorbar(layer_names, means, yerr=stds, fmt="o-", capsize=5)
    plt.xlabel("Layer")
    plt.ylabel("Mean Entropy")
    plt.title("Layer-wise Entropy (Mean ± Std across Folds)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_distribution_mean.pdf")
    plt.close()


def plot_representative_activations(model, output_dir):
    layers = extract_actmix_layers(model)
    for layer_name, layer in layers:
        safe_name = layer_name.replace(".", "_")
        coefficients = get_mixing_coefficients(layer)
        entropy = get_entropy_per_neuron(layer)
        basis_names = layer.basis_function_names
        sorted_indices = np.argsort(entropy)
        selected_indices = np.concatenate([sorted_indices[:3], sorted_indices[-3:]])
        x = torch.linspace(-5, 5, 200)
        plt.figure(figsize=(12, 8))
        for neuron_idx in selected_indices:
            coeffs = torch.from_numpy(coefficients[neuron_idx])
            y = torch.zeros_like(x)
            for k, name in enumerate(basis_names):
                if name == "sin":
                    basis_y = torch.sin(layer.omega_0 * x)
                elif name == "tanh":
                    basis_y = torch.tanh(x)
                elif name == "relu":
                    basis_y = F.relu(x)
                elif name == "identity":
                    basis_y = x
                elif name == "sigmoid":
                    basis_y = torch.sigmoid(x)
                elif name == "silu":
                    basis_y = F.silu(x)
                else:
                    basis_y = x
                y += coeffs[k] * basis_y
            label_type = (
                "High Entropy" if neuron_idx in sorted_indices[-3:] else "Low Entropy"
            )
            plt.plot(x.numpy(), y.numpy(), label=f"N{neuron_idx} ({label_type})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(
            f"Effective Activation Functions - {layer_name} (Representative Fold)"
        )
        plt.xlabel("Input (x)")
        plt.ylabel("Output (y)")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{safe_name}_effective_activations_representative.pdf"
        )
        plt.close()


def main():
    datasets = ["adult_census", "magic_gamma", "power_plant"]
    architecture = "1x128-full"
    for dataset in datasets:
        print(f"Processing {dataset}...")
        base_path = (
            project_root
            / "models"
            / architecture
            / "experiments"
            / f"mlp_actmix-{dataset}"
            / "checkpoints"
        )
        all_fold_stats = []
        representative_model = None
        found_folds = 0
        for fold in range(10):
            fold_path = base_path / f"fold_{fold}"
            checkpoints = list(fold_path.glob("best*.ckpt"))
            if not checkpoints:
                checkpoints = list(fold_path.glob("last.ckpt"))
            if not checkpoints:
                continue
            ckpt_path = checkpoints[0]
            try:
                model = load_model_from_checkpoint(ckpt_path)
                model.eval()
                layers = extract_actmix_layers(model)
                fold_stats = {}
                for layer_name, layer in layers:
                    fold_stats[layer_name] = compute_layer_statistics(layer)
                all_fold_stats.append(fold_stats)
                found_folds += 1
                if fold == 0:
                    representative_model = model
            except Exception as e:
                print(f"Error processing fold {fold}: {e}")
        if not all_fold_stats:
            print(f"No data found for {dataset}")
            continue
        print(f"Aggregating over {found_folds} folds for {dataset}")
        aggregated = aggregate_statistics(all_fold_stats)
        output_dir = project_root / "paper" / "figures" / f"analysis_{dataset}"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_aggregated_topology(aggregated, output_dir)
        plot_aggregated_specialization(aggregated, output_dir)
        plot_aggregated_entropy(aggregated, output_dir)
        if representative_model:
            plot_representative_activations(representative_model, output_dir)

        def convert(o):
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError

        with open(output_dir / "aggregated_statistics.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=convert)


if __name__ == "__main__":
    main()
