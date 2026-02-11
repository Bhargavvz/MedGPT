#!/bin/bash
# ============================================================================
# Medical VQA - H200 Training Launch Script
# ============================================================================
set -euo pipefail

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  KGET Medical VQA - H200 Training${NC}"
echo -e "${CYAN}================================================================${NC}"

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate medvqa

# Source environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# H200 optimal environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TORCH_CUDA_ARCH_LIST="9.0"

# Print GPU info
echo ""
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'bf16: {torch.cuda.is_bf16_supported()}')
print(f'Arch: {torch.cuda.get_device_capability()}')
"
echo ""

# Parse arguments
CONFIG="${1:-config/h200_config.yaml}"
EXTRA_ARGS="${@:2}"

echo -e "${GREEN}Config: ${CONFIG}${NC}"
echo -e "${GREEN}Extra Args: ${EXTRA_ARGS}${NC}"
echo ""

# ============================================================================
# Option 1: Standard Training (single GPU)
# ============================================================================
echo -e "${CYAN}Starting training...${NC}"

python training/train.py \
    --config "$CONFIG" \
    --output_dir ./checkpoints \
    --logging_dir ./logs \
    $EXTRA_ARGS

echo ""
echo -e "${GREEN}Training complete!${NC}"
echo -e "${GREEN}Checkpoints: ./checkpoints/${NC}"
echo -e "${GREEN}Logs: ./logs/${NC}"
