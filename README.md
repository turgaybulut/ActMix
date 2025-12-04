# ActMix: Recovering Tabular Inductive Bias via Adaptive Activation Topology

This project implements **ActMix**, a novel deep learning architecture designed to recover tabular inductive bias through adaptive activation topology. It includes implementations of ActMix layers, standard MLP baselines, and training/analysis scripts for various UCI datasets.

## Project Structure

```bash
ActMix/
├── conf/                   # Hydra configuration files
│   ├── dataset/            # Dataset specific configs (adult_census, power_plant, etc.)
│   ├── model/              # Model specific configs (actmix, mlp_relu, etc.)
│   └── training/           # Training hyperparameters
├── scripts/                # Executable scripts
│   ├── train.py            # Main training script
│   └── analyze_weights.py  # Weight analysis and visualization script
├── src/                    # Source code
│   ├── datamodules/        # PyTorch Lightning DataModules
│   ├── models/             # Model architectures and layers
│   └── system.py           # LightningModule system definition
└── pyproject.toml          # Project dependencies
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

- Python 3.9+ installed
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Install the project

```bash
# Clone the repository
git clone git@github.com:turgaybulut/ActMix.git
cd ActMix

# Install the project and its dependencies
uv sync

# For development
uv sync --group dev
```

## Usage

### Training

The project uses [Hydra](https://hydra.cc/) for configuration management. You can train models by running `scripts/train.py` and overriding configuration groups.

**Train ActMix on the Power Plant dataset (default):**

```bash
python scripts/train.py
```

**Train ActMix on the Adult Census dataset:**

```bash
python scripts/train.py dataset=adult_census model=actmix
```

**Train a ReLU MLP baseline on the Magic Gamma dataset:**

```bash
python scripts/train.py dataset=magic_gamma model=mlp_relu
```

**Common overrides:**

- `dataset`: `power_plant`, `adult_census`, `magic_gamma`
- `model`: `actmix`, `mlp_relu`, `mlp_gelu`, `mlp_prelu`
- `seed`: Random seed (default: 1192)

**Train all supported models for a dataset (works from any working directory):**

```bash
./scripts/train_all_models.sh [dataset]
```

The dataset argument accepts `power_plant`, `adult_census`, or `magic_gamma` and defaults to `power_plant`.

### Analysis

After training, you can analyze the learned mixing coefficients and entropy of ActMix layers using the `analyze_weights.py` script. This generates visualizations and a statistical report.

```bash
python scripts/analyze_weights.py \
    --checkpoint experiments/actmix_power_plant/checkpoints/last.ckpt \
    --output_dir analysis_results
```

**Arguments:**

- `--checkpoint`: Path to the trained model checkpoint (`.ckpt` file).
- `--output_dir`: Directory to save the analysis report and plots (default: `analysis_output`).

## Development

This project uses modern Python tooling with uv for dependency management and ruff for code formatting and linting.

### Code Quality

Format code with ruff:

```bash
uv run ruff format .
```

Lint code:

```bash
uv run ruff check .
```

### Running Scripts

Use uv to run scripts with the correct environment:

```bash
# Training
uv run python scripts/train.py

# Analysis
uv run python scripts/analyze_weights.py --checkpoint path/to/checkpoint.ckpt
```
