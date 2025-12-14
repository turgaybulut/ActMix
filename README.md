# ActMix: Recovering Tabular Inductive Bias via Adaptive Activation Topology

ActMix is a novel deep learning architecture designed to recover tabular inductive bias through adaptive activation topology. Unlike static activation functions that impose homogeneous geometric transformations, ActMix enables each neuron to learn its own convex combination of topologically distinct basis functions (Sine, Tanh, ReLU, Identity).

## Key Features

- **Neuron-wise Activation Mixing**: Each neuron learns its own weighted combination of basis functions
- **Temperature Annealing**: Softmax temperature decreases during training to sharpen activation selection
- **Entropy Regularization**: Encourages layer-wise specialization to prevent mean-field collapse
- **Interpretable Weights**: Visualize which activation functions each neuron prefers

## Project Structure

```text
ActMix/
├── conf/                       # Hydra configuration files
│   ├── config.yaml             # Main configuration
│   ├── dataset/                # Dataset configs (power_plant, adult_census, magic_gamma)
│   ├── model/                  # Model configs (mlp_actmix, mlp_relu, mlp_gelu, mlp_prelu)
│   └── training/               # Training hyperparameters
├── scripts/                    # Executable scripts
│   ├── train.py                # Main training script
│   ├── train_cv.py             # Cross-validation training script
│   ├── analyze_weights.py      # Weight analysis and visualization
│   ├── benchmark_xgboost.py    # XGBoost baseline benchmark
│   └── train_cv_all_models.sh  # Batch Cross-Validation script
├── src/                        # Source code
│   ├── config.py               # Structured configuration dataclasses
│   ├── metrics.py              # Metric computation utilities
│   ├── schedulers.py           # Temperature and entropy schedulers
│   ├── trainer.py              # PyTorch Lightning TabularTrainer module
│   ├── datamodules/            # Data loading infrastructure
│   │   ├── base.py             # Abstract base classes
│   │   ├── uci_classification.py
│   │   └── uci_regression.py
│   └── models/                 # Model architectures
│       ├── base.py             # Protocols and StaticMLP baseline
│       ├── actmix_mlp.py       # ActMix model
│       └── layers/
│           └── actmix_layer.py # Core ActMix layer implementation
└── pyproject.toml              # Project dependencies
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install

```bash
git clone git@github.com:turgaybulut/ActMix.git
cd ActMix

# Install dependencies
uv sync

# For development (includes ruff)
uv sync --group dev
```

## Usage

### Training

The project uses [Hydra](https://hydra.cc/) for configuration management.

**Train ActMix on Power Plant (default):**

```bash
uv run python scripts/train.py
```

**Train on different datasets:**

```bash
# Adult Census (classification)
uv run python scripts/train.py dataset=adult_census model=mlp_actmix

# Magic Gamma (classification)
uv run python scripts/train.py dataset=magic_gamma model=mlp_actmix
```

**Train baseline models:**

```bash
# ReLU MLP
uv run python scripts/train.py model=mlp_relu

# GELU MLP
uv run python scripts/train.py model=mlp_gelu

# PReLU MLP
uv run python scripts/train.py model=mlp_prelu
```

**Common configuration overrides:**

| Parameter | Options | Default |
|-----------|---------|---------|
| `dataset` | `power_plant`, `adult_census`, `magic_gamma` | `power_plant` |
| `model` | `mlp_actmix`, `mlp_relu`, `mlp_gelu`, `mlp_prelu` | `mlp_actmix` |
| `seed` | Any integer | `1192` |
| `training.max_epochs` | Any integer | `200` |
| `training.learning_rate` | Any float | `1e-3` |

**Run Cross-Validation benchmark for all models:**

```bash
# Default (10 folds)
./scripts/train_cv_all_models.sh power_plant

# Custom number of folds
./scripts/train_cv_all_models.sh adult_census 5
./scripts/train_cv_all_models.sh magic_gamma 10
```

### Cross-Validation

Run k-fold cross-validation for robust model evaluation:

```bash
# 10-fold CV (default)
uv run python scripts/train_cv.py dataset=power_plant model=mlp_actmix

# Custom number of folds
uv run python scripts/train_cv.py dataset=magic_gamma model=mlp_actmix cv_folds=5

# Quick test with reduced epochs
uv run python scripts/train_cv.py training.max_epochs=5 dataset=power_plant model=mlp_actmix cv_folds=3
```

**Features:**

- StratifiedKFold for classification (preserves class distribution)
- KFold for regression tasks
- Aggregated results table with mean ± std, min, max
- Per-fold results saved to `experiments/<model>_<dataset>/cv_results.json`

### Analysis

Analyze learned mixing coefficients after training:

```bash
uv run python scripts/analyze_weights.py \
    --checkpoint experiments/mlp_actmix_power_plant/checkpoints/last.ckpt \
    --output_dir analysis_results
```

This generates:

- Mixing coefficient heatmaps
- Entropy distribution plots
- Dominant basis function pie charts
- Per-neuron coefficient bar charts
- Effective activation function plots (High vs Low Entropy)
- Layer-wise topology evolution
- Neuron specialization spectrum
- `layer_statistics.json` with numerical summaries

### XGBoost Benchmark

Run XGBoost baseline (10-fold Cross-Validation) on all datasets:

```bash
uv run python scripts/benchmark_xgboost.py
```

Results (Mean ± Std) saved to `experiments/xgboost_results/results.json`.

## Model Architecture

### ActMix Layer

The core innovation replaces static activations with a learnable mixture:

```text
ActMix(z) = Σ αₖ φₖ(z)
```

where:

- `φₖ ∈ {sin, tanh, relu, identity}` are basis functions
- `αₖ = softmax(aₖ / T)` are learned mixing coefficients
- `T` is temperature (annealed from 1.0 to 0.01 during training)

### Configuration Options

**ActMix-specific parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `basis_functions` | List of basis functions | `[sin, tanh, relu, identity]` |
| `relu_bias` | Initial bias for ReLU in mixing logits | `1.5` |
| `omega_0` | Frequency scaling for sine activation | `1.0` |
| `temperature.initial` | Starting temperature | `1.0` |
| `temperature.final` | Final temperature | `0.01` |
| `temperature.anneal_epochs` | Epochs to anneal | `150` |
| `entropy_regularization.lambda_max` | Max entropy penalty | `0.01` |
| `entropy_regularization.warmup_epochs` | Warmup epochs | `50` |

## Datasets

| Dataset | Task | Features | Classes/Target |
|---------|------|----------|----------------|
| Power Plant | Regression | 4 | 1 (energy output) |
| Adult Census | Classification | 14 | 2 (income >50K) |
| Magic Gamma | Classification | 10 | 2 (gamma/hadron) |

All datasets are automatically downloaded from the UCI Machine Learning Repository.

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix
```

### Running Scripts

Always use `uv run` to ensure correct environment:

```bash
uv run python scripts/train.py
uv run python scripts/train_cv.py
uv run python scripts/analyze_weights.py --checkpoint path/to/checkpoint.ckpt
uv run python scripts/benchmark_xgboost.py
```

## License

MIT License
