import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import OmegaConf

from src.models import ActMixMLP
from src.models.layers import ActMixLayer
from src.system import ActMixSystem


def load_config(config_path: Path):
    return OmegaConf.load(config_path)


def create_model_from_config(config) -> ActMixMLP:
    model_conf = config.model
    dataset_conf = config.dataset

    return ActMixMLP(
        input_dim=dataset_conf.input_dim,
        output_dim=dataset_conf.output_dim,
        hidden_dims=list(model_conf.hidden_dims),
        basis_functions=list(model_conf.basis_functions),
        dropout_rate=model_conf.get("dropout_rate", 0.0),
        relu_bias=model_conf.get("relu_bias", 1.5),
    )


def load_model_from_checkpoint(checkpoint_path: str) -> ActMixMLP:
    chkpt_path = Path(checkpoint_path)
    exp_dir = chkpt_path.parent.parent
    config_path = exp_dir / "logs" / ".hydra" / "config.yaml"

    assert config_path.exists()

    config = load_config(config_path)
    model = create_model_from_config(config)

    system = ActMixSystem.load_from_checkpoint(checkpoint_path, model=model)
    return system.model


def extract_actmix_layers(model: ActMixMLP) -> list[tuple[str, ActMixLayer]]:
    actmix_layers = []
    for name, module in model.named_modules():
        if isinstance(module, ActMixLayer):
            actmix_layers.append((name, module))
    return actmix_layers


def get_mixing_coefficients(layer: ActMixLayer) -> np.ndarray:
    with torch.no_grad():
        coefficients = layer.get_mixing_coefficients()
        return coefficients.cpu().numpy()


def get_entropy_per_neuron(layer: ActMixLayer) -> np.ndarray:
    with torch.no_grad():
        coefficients = layer.get_mixing_coefficients()
        entropy = -torch.sum(coefficients * torch.log(coefficients + 1e-10), dim=-1)
        return entropy.cpu().numpy()


def compute_layer_statistics(layer: ActMixLayer) -> dict[str, float]:
    coefficients = get_mixing_coefficients(layer)
    entropy = get_entropy_per_neuron(layer)

    dominant_indices = np.argmax(coefficients, axis=-1)
    basis_function_names = layer.basis_function_names

    dominance_counts = {}
    for name in basis_function_names:
        dominance_counts[name] = 0

    for idx in dominant_indices:
        name = basis_function_names[idx]
        dominance_counts[name] += 1

    return {
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "min_entropy": float(np.min(entropy)),
        "max_entropy": float(np.max(entropy)),
        "dominance_counts": dominance_counts,
        "num_neurons": len(dominant_indices),
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
    max_neurons: int = 20,
) -> None:
    coefficients = get_mixing_coefficients(layer)
    basis_names = layer.basis_function_names

    num_neurons_to_plot = min(max_neurons, coefficients.shape[0])

    fig, axes = plt.subplots(
        num_neurons_to_plot,
        1,
        figsize=(8, 2 * num_neurons_to_plot),
        sharex=True,
    )

    if num_neurons_to_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(basis_names, coefficients[i])
        ax.set_ylabel(f"N{i}")
        ax.set_ylim(0, 1)

    plt.xlabel("Basis Function")
    fig.suptitle(f"Mixing Coefficients per Neuron - {layer_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_coefficients_bar.png", dpi=150)
    plt.close()


def generate_report(
    model: ActMixMLP,
    output_dir: Path,
) -> str:
    layers = extract_actmix_layers(model)

    report_lines = ["# ActMix Weight Analysis Report", ""]

    for layer_name, layer in layers:
        statistics = compute_layer_statistics(layer)

        report_lines.append(f"## Layer: {layer_name}")
        report_lines.append("")
        report_lines.append(f"- Number of neurons: {statistics['num_neurons']}")
        report_lines.append(f"- Mean entropy: {statistics['mean_entropy']:.4f}")
        report_lines.append(f"- Std entropy: {statistics['std_entropy']:.4f}")
        report_lines.append(f"- Min entropy: {statistics['min_entropy']:.4f}")
        report_lines.append(f"- Max entropy: {statistics['max_entropy']:.4f}")
        report_lines.append("")
        report_lines.append("### Dominant Basis Function Counts:")
        for name, count in statistics["dominance_counts"].items():
            percentage = 100 * count / statistics["num_neurons"]
            report_lines.append(f"  - {name}: {count} ({percentage:.1f}%)")
        report_lines.append("")

        plot_mixing_coefficients_heatmap(
            layer, layer_name.replace(".", "_"), output_dir
        )
        plot_entropy_distribution(layer, layer_name.replace(".", "_"), output_dir)
        plot_dominance_pie_chart(statistics, layer_name.replace(".", "_"), output_dir)
        plot_coefficient_bar_per_neuron(layer, layer_name.replace(".", "_"), output_dir)

    return "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint(args.checkpoint)
    model.eval()

    report = generate_report(model, output_dir)

    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Analysis complete. Results saved to {output_dir}")
    print("\nReport Summary:")
    print(report)


if __name__ == "__main__":
    main()
