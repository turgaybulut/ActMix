import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.datamodules import DATAMODULE_REGISTRY
from src.models import ActMixMLP
from src.models.layers import ActMixLayer
from src.trainer import TabularTrainer


def load_config(config_path: Path) -> dict:
    return OmegaConf.load(config_path)


def create_model_from_config(config: dict) -> ActMixMLP:
    model_conf = config.model
    dataset_conf = config.dataset

    return ActMixMLP(
        input_dim=dataset_conf.input_dim,
        output_dim=dataset_conf.output_dim,
        hidden_dims=list(model_conf.hidden_dims),
        basis_functions=list(model_conf.basis_functions),
        dropout_rate=model_conf.get("dropout_rate", 0.0),
        relu_bias=model_conf.get("relu_bias", 1.5),
        omega_0=model_conf.get("omega_0", 1.0),
    )


def get_datamodule_dims(config: dict) -> tuple[int, int]:
    datamodule_cls = DATAMODULE_REGISTRY[config.dataset.name]
    datamodule = datamodule_cls(
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        seed=config.seed,
        pin_memory=config.dataset.pin_memory,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule.input_dim, datamodule.output_dim


def load_model_from_checkpoint(checkpoint_path: Path) -> ActMixMLP:
    current_dir = checkpoint_path.parent
    config_path = None

    for _ in range(4):
        possible_config = current_dir / "config.yaml"
        if possible_config.exists():
            config_path = possible_config
            break

        possible_hydra_config = current_dir / "logs" / ".hydra" / "config.yaml"
        if possible_hydra_config.exists():
            config_path = possible_hydra_config
            break

        current_dir = current_dir.parent

        if current_dir == current_dir.parent:
            break

    assert config_path is not None and config_path.exists(), (
        f"Config not found strictly above {checkpoint_path}"
    )

    config = load_config(config_path)
    config.dataset.input_dim, config.dataset.output_dim = get_datamodule_dims(config)

    model = create_model_from_config(config)

    trainer = TabularTrainer.load_from_checkpoint(str(checkpoint_path), model=model)
    return trainer.model


def extract_actmix_layers(model: ActMixMLP) -> list[tuple[str, ActMixLayer]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ActMixLayer)
    ]


def get_mixing_coefficients(layer: ActMixLayer) -> np.ndarray:
    with torch.no_grad():
        return layer.get_mixing_coefficients().cpu().numpy()


def get_entropy_per_neuron(layer: ActMixLayer) -> np.ndarray:
    with torch.no_grad():
        coefficients = layer.get_mixing_coefficients()
        log_coefficients = torch.log(coefficients + 1e-10)
        entropy = -torch.sum(coefficients * log_coefficients, dim=-1)
        return entropy.cpu().numpy()


def compute_layer_statistics(layer: ActMixLayer) -> dict[str, float | dict | list]:
    coefficients = get_mixing_coefficients(layer)
    entropy = get_entropy_per_neuron(layer)

    dominant_indices = np.argmax(coefficients, axis=-1)
    basis_names = layer.basis_function_names

    dominance_counts = {name: 0 for name in basis_names}
    for idx in dominant_indices:
        dominance_counts[basis_names[idx]] += 1

    return {
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "min_entropy": float(np.min(entropy)),
        "max_entropy": float(np.max(entropy)),
        "dominance_counts": dominance_counts,
        "num_neurons": len(dominant_indices),
        "basis_mass": np.sum(coefficients, axis=0).tolist(),
        "max_alphas": np.max(coefficients, axis=-1).tolist(),
    }


def plot_mixing_coefficients_heatmap(
    layer: ActMixLayer,
    layer_name: str,
    output_dir: Path,
) -> None:
    coefficients = get_mixing_coefficients(layer)
    basis_names = layer.basis_function_names

    plt.figure(figsize=(12, max(4, coefficients.shape[0] // 10)))
    sns.heatmap(
        coefficients,
        xticklabels=basis_names,
        yticklabels=False,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Mixing Coefficient"},
    )
    plt.xlabel("Basis Function")
    plt.ylabel("Neuron Index")
    plt.title(f"Mixing Coefficients - {layer_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_coefficients_heatmap.png", dpi=150)
    plt.close()


def plot_entropy_distribution(
    layer: ActMixLayer,
    layer_name: str,
    output_dir: Path,
) -> None:
    entropy = get_entropy_per_neuron(layer)
    max_entropy = np.log(len(layer.basis_function_names))

    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(
        max_entropy,
        color="red",
        linestyle="--",
        label=f"Max Entropy ({max_entropy:.3f})",
    )
    plt.axvline(
        np.mean(entropy),
        color="green",
        linestyle="--",
        label=f"Mean ({np.mean(entropy):.3f})",
    )
    plt.xlabel("Entropy")
    plt.ylabel("Number of Neurons")
    plt.title(f"Entropy Distribution - {layer_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_entropy_distribution.png", dpi=150)
    plt.close()


def plot_dominance_pie_chart(
    statistics: dict,
    layer_name: str,
    output_dir: Path,
) -> None:
    dominance_counts = statistics["dominance_counts"]
    labels = list(dominance_counts.keys())
    sizes = list(dominance_counts.values())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(f"Dominant Basis Function Distribution - {layer_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_dominance_pie.png", dpi=150)
    plt.close()


def plot_coefficient_bar_per_neuron(
    layer: ActMixLayer,
    layer_name: str,
    output_dir: Path,
) -> None:
    coefficients = get_mixing_coefficients(layer)
    entropy = get_entropy_per_neuron(layer)
    basis_names = layer.basis_function_names

    sorted_indices = np.argsort(entropy)
    selected_indices = np.concatenate([sorted_indices[:3], sorted_indices[-3:]])

    num_plots = len(selected_indices)
    fig, axes = plt.subplots(
        num_plots,
        1,
        figsize=(8, 2 * num_plots),
        sharex=True,
    )

    for i, neuron_idx in enumerate(selected_indices):
        ax = axes[i]
        ax.bar(basis_names, coefficients[neuron_idx])

        label_type = (
            "Low Entropy" if neuron_idx in sorted_indices[:3] else "High Entropy"
        )
        ax.set_ylabel(f"N{neuron_idx}\n({label_type})")
        ax.set_ylim(0, 1)

    plt.xlabel("Basis Function")
    fig.suptitle(f"Mixing Coefficients (Extreme Entropy Neurons) - {layer_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_coefficients_bar.png", dpi=150)
    plt.close()


def plot_effective_activation_functions(
    layer: ActMixLayer,
    layer_name: str,
    output_dir: Path,
) -> None:
    coefficients = get_mixing_coefficients(layer)
    entropy = get_entropy_per_neuron(layer)
    basis_names = layer.basis_function_names

    sorted_indices = np.argsort(entropy)
    lowest_entropy_indices = sorted_indices[:3]
    highest_entropy_indices = sorted_indices[-3:]
    selected_indices = np.concatenate([lowest_entropy_indices, highest_entropy_indices])

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
            "High Entropy" if neuron_idx in highest_entropy_indices else "Low Entropy"
        )
        plt.plot(x.numpy(), y.numpy(), label=f"N{neuron_idx} ({label_type})")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Effective Activation Functions - {layer_name}")
    plt.xlabel("Input (x)")
    plt.ylabel("Output (y)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_effective_activations.png", dpi=150)
    plt.close()


def plot_layer_topology_evolution(
    all_statistics: dict,
    output_dir: Path,
) -> None:
    layer_names = list(all_statistics.keys())
    basis_names = list(all_statistics[layer_names[0]]["dominance_counts"].keys())

    mass_data = {name: [] for name in basis_names}

    for layer in layer_names:
        mass = all_statistics[layer]["basis_mass"]
        for i, name in enumerate(basis_names):
            mass_data[name].append(mass[i])

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(layer_names))

    for name in basis_names:
        values = np.array(mass_data[name])
        ax.bar(layer_names, values, bottom=bottom, label=name)
        bottom += values

    plt.xlabel("Layer")
    plt.ylabel("Total Mixing Coefficient Mass")
    plt.title("Layer-wise Topology Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "layer_topology_evolution.png", dpi=150)
    plt.close()


def plot_specialization_spectrum(
    all_max_alphas: list[float],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(all_max_alphas, bins=50, range=(0, 1), edgecolor="black", alpha=0.7)
    plt.axvline(0.25, color="red", linestyle="--", label="Mean-Field (0.25)")
    plt.axvline(1.0, color="green", linestyle="--", label="Specialized (1.0)")

    plt.xlabel("Max Mixing Coefficient")
    plt.ylabel("Count")
    plt.title("Neuron Specialization Spectrum (All Layers)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "specialization_spectrum.png", dpi=150)
    plt.close()


def generate_report(model: ActMixMLP, output_dir: Path) -> dict:
    layers = extract_actmix_layers(model)
    all_statistics = {}
    all_max_alphas = []

    for layer_name, layer in layers:
        safe_name = layer_name.replace(".", "_")
        statistics = compute_layer_statistics(layer)
        all_statistics[layer_name] = statistics
        all_max_alphas.extend(statistics["max_alphas"])

        plot_mixing_coefficients_heatmap(layer, safe_name, output_dir)
        plot_entropy_distribution(layer, safe_name, output_dir)
        plot_dominance_pie_chart(statistics, safe_name, output_dir)
        plot_coefficient_bar_per_neuron(layer, safe_name, output_dir)
        plot_effective_activation_functions(layer, safe_name, output_dir)

    plot_layer_topology_evolution(all_statistics, output_dir)
    plot_specialization_spectrum(all_max_alphas, output_dir)

    return all_statistics


def save_statistics_json(statistics: dict, output_dir: Path) -> None:
    output_file = output_dir / "layer_statistics.json"
    with open(output_file, "w") as f:
        json.dump(statistics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    model = load_model_from_checkpoint(checkpoint_path)
    model.eval()

    statistics = generate_report(model, output_dir)
    save_statistics_json(statistics, output_dir)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
