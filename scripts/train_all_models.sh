#!/bin/bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

show_usage() {
    echo -e "${CYAN}Usage: $0 [dataset]${NC}"
    echo -e "${BLUE}Available datasets: power_plant, adult_census, magic_gamma${NC}"
    echo -e "${BLUE}Default dataset: power_plant${NC}"
    echo -e "${YELLOW}Example: $0 power_plant${NC}"
    echo -e "${YELLOW}Example: $0 adult_census${NC}"
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

DATASET="${1:-power_plant}"

VALID_DATASETS=("power_plant" "adult_census" "magic_gamma")
if [[ ! " ${VALID_DATASETS[@]} " =~ " ${DATASET} " ]]; then
    echo -e "${RED}Error: Invalid dataset '$DATASET'${NC}"
    echo -e "${BLUE}Valid datasets: ${VALID_DATASETS[*]}${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

MODELS=("actmix" "mlp_relu" "mlp_gelu" "mlp_prelu")

echo -e "${CYAN}Starting training pipeline for all models with $DATASET dataset${NC}"
echo -e "${BLUE}Models to train: ${MODELS[*]}${NC}"
echo -e "${YELLOW}=======================================================================================${NC}"

for model in "${MODELS[@]}"; do
    echo -e "${BLUE}Running ${YELLOW}$model${BLUE} with $DATASET dataset...${NC}"

    uv run python "$PROJECT_ROOT/scripts/train.py" \
        dataset="$DATASET" \
        model="$model" \
        seed=1192

    echo -e "${GREEN}Completed $model training.${NC}"
    echo -e "${YELLOW}=======================================================================================${NC}"
done

echo -e "${GREEN}All models trained successfully with $DATASET dataset.${NC}"