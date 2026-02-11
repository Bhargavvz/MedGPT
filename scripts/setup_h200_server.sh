#!/bin/bash
# ============================================================================
# Medical VQA - H200 GPU Server Setup Script
# Knowledge-Guided Explainable Transformer
# ============================================================================
# Server: NVIDIA H200 GPU (141GB HBM3e) + 1.9TB RAM
# OS: Ubuntu 22.04 (assumed)
# ============================================================================
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${CYAN}================================================================${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}================================================================${NC}"; }

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR="${1:-/workspace/MedicalVQA}"
CONDA_ENV="medvqa"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.4"  # H200 uses CUDA 12.x

# Model HuggingFace IDs
BASE_MODEL="Qwen/Qwen2-VL-7B-Instruct"
KNOWLEDGE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
VISION_MODEL="openai/clip-vit-large-patch14"
BIOBERT_MODEL="dmis-lab/biobert-base-cased-v1.2"

# Dataset URLs
VQARAD_URL="https://huggingface.co/datasets/flaviagiammarino/vqa-rad"
SLAKE_URL="https://huggingface.co/datasets/BoKelvin/SLAKE"
PATHVQA_URL="https://huggingface.co/datasets/flaviagiammarino/path-vqa"

log_step "Medical VQA - H200 Server Setup"
echo ""
echo "  Project Directory: ${PROJECT_DIR}"
echo "  Python Version:    ${PYTHON_VERSION}"
echo "  CUDA Version:      ${CUDA_VERSION}"
echo "  GPU:               NVIDIA H200 (141GB HBM3e)"
echo "  RAM:               1.9 TB"
echo ""

# ============================================================================
# Step 1: System Dependencies
# ============================================================================
log_step "Step 1: Installing System Dependencies"

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    2>/dev/null

git lfs install

log_info "System dependencies installed."

# ============================================================================
# Step 2: Setup Conda Environment
# ============================================================================
log_step "Step 2: Setting Up Python Environment"

# Install miniconda if not available
if ! command -v conda &> /dev/null; then
    log_info "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
fi

# Create conda environment
if conda info --envs | grep -q "$CONDA_ENV"; then
    log_warn "Conda environment '${CONDA_ENV}' already exists. Activating..."
else
    log_info "Creating conda environment '${CONDA_ENV}'..."
    conda create -n "$CONDA_ENV" python="${PYTHON_VERSION}" -y
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

log_info "Python environment ready: $(python --version)"

# ============================================================================
# Step 3: Install PyTorch with CUDA 12.4 (H200)
# ============================================================================
log_step "Step 3: Installing PyTorch for H200 GPU"

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.4 for H200
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'bf16 Support: {torch.cuda.is_bf16_supported()}')
"

log_info "PyTorch installed with CUDA support."

# ============================================================================
# Step 4: Install Flash Attention 2
# ============================================================================
log_step "Step 4: Installing Flash Attention 2"

pip install flash-attn --no-build-isolation

python -c "
import flash_attn
print(f'Flash Attention Version: {flash_attn.__version__}')
" && log_info "Flash Attention 2 installed." || log_warn "Flash Attention install may need manual build."

# ============================================================================
# Step 5: Install Project Dependencies
# ============================================================================
log_step "Step 5: Installing Project Dependencies"

# Clone or navigate to project
if [ ! -d "$PROJECT_DIR" ]; then
    log_info "Creating project directory..."
    mkdir -p "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# Install main requirements
pip install -r requirements.txt

# Install additional H200-optimized packages
pip install \
    xformers \
    triton \
    ninja \
    packaging \
    wandb \
    gradio \
    vllm

# Install SciSpacy model for medical NER
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

log_info "All Python dependencies installed."

# ============================================================================
# Step 6: Download AI Models
# ============================================================================
log_step "Step 6: Downloading AI Models"

MODELS_DIR="${PROJECT_DIR}/models_cache"
mkdir -p "$MODELS_DIR"
export HF_HOME="$MODELS_DIR"
export TRANSFORMERS_CACHE="$MODELS_DIR"

log_info "Downloading Qwen2-VL-7B-Instruct (~15GB)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${BASE_MODEL}',
    cache_dir='${MODELS_DIR}',
    resume_download=True,
)
print('Qwen2-VL-7B-Instruct downloaded.')
"

log_info "Downloading PubMedBERT (~440MB)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${KNOWLEDGE_MODEL}',
    cache_dir='${MODELS_DIR}',
    resume_download=True,
)
print('PubMedBERT downloaded.')
"

log_info "Downloading CLIP ViT-Large (~1.7GB)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${VISION_MODEL}',
    cache_dir='${MODELS_DIR}',
    resume_download=True,
)
print('CLIP ViT-Large downloaded.')
"

log_info "Downloading BioBERT (~440MB)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${BIOBERT_MODEL}',
    cache_dir='${MODELS_DIR}',
    resume_download=True,
)
print('BioBERT downloaded.')
"

log_info "All AI models downloaded to ${MODELS_DIR}"

# ============================================================================
# Step 7: Download Datasets
# ============================================================================
log_step "Step 7: Downloading Datasets"

DATA_DIR="${PROJECT_DIR}/data/raw"
mkdir -p "$DATA_DIR/vqa_rad" "$DATA_DIR/slake" "$DATA_DIR/pathvqa"

log_info "Downloading VQA-RAD dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('flaviagiammarino/vqa-rad', cache_dir='${DATA_DIR}/vqa_rad')
ds.save_to_disk('${DATA_DIR}/vqa_rad/dataset')
print(f'VQA-RAD: {ds}')
"

log_info "Downloading SLAKE dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('BoKelvin/SLAKE', cache_dir='${DATA_DIR}/slake')
ds.save_to_disk('${DATA_DIR}/slake/dataset')
print(f'SLAKE: {ds}')
"

log_info "Downloading PathVQA dataset..."
python -c "
from datasets import load_dataset
ds = load_dataset('flaviagiammarino/path-vqa', cache_dir='${DATA_DIR}/pathvqa')
ds.save_to_disk('${DATA_DIR}/pathvqa/dataset')
print(f'PathVQA: {ds}')
"

log_info "All datasets downloaded to ${DATA_DIR}"

# ============================================================================
# Step 8: Prepare Datasets into Unified Format
# ============================================================================
log_step "Step 8: Preparing Unified Dataset"

python -c "
import json
import os
from datasets import load_from_disk
from pathlib import Path

DATA_DIR = '${DATA_DIR}'
OUTPUT_DIR = '${PROJECT_DIR}/data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

unified_data = {'train': [], 'val': [], 'test': []}

# ---- VQA-RAD ----
print('Processing VQA-RAD...')
try:
    ds = load_from_disk(f'{DATA_DIR}/vqa_rad/dataset')
    for split_name, target in [('train', 'train'), ('test', 'test')]:
        if split_name in ds:
            for i, item in enumerate(ds[split_name]):
                entry = {
                    'id': f'vqarad_{split_name}_{i}',
                    'dataset': 'VQA-RAD',
                    'image_path': f'vqa_rad/images/{split_name}_{i}.jpg',
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'answer_type': item.get('answer_type', 'OPEN'),
                    'question_type': item.get('question_type', 'other'),
                }
                unified_data[target].append(entry)
                # Save image
                img_dir = f'{DATA_DIR}/vqa_rad/images'
                os.makedirs(img_dir, exist_ok=True)
                if 'image' in item and item['image'] is not None:
                    img_path = f'{img_dir}/{split_name}_{i}.jpg'
                    if not os.path.exists(img_path):
                        item['image'].save(img_path)
    print(f'  VQA-RAD: {sum(len(v) for v in unified_data.values())} samples processed')
except Exception as e:
    print(f'  VQA-RAD error: {e}')

# ---- SLAKE ----
print('Processing SLAKE...')
try:
    ds = load_from_disk(f'{DATA_DIR}/slake/dataset')
    for split_name in ['train', 'validation', 'test']:
        target = 'val' if split_name == 'validation' else split_name
        if split_name in ds:
            for i, item in enumerate(ds[split_name]):
                entry = {
                    'id': f'slake_{split_name}_{i}',
                    'dataset': 'SLAKE',
                    'image_path': f'slake/images/{split_name}_{i}.jpg',
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'answer_type': item.get('answer_type', 'OPEN'),
                    'question_type': item.get('q_lang', 'en'),
                }
                unified_data[target].append(entry)
                img_dir = f'{DATA_DIR}/slake/images'
                os.makedirs(img_dir, exist_ok=True)
                if 'image' in item and item['image'] is not None:
                    img_path = f'{img_dir}/{split_name}_{i}.jpg'
                    if not os.path.exists(img_path):
                        item['image'].save(img_path)
    print(f'  SLAKE: processed')
except Exception as e:
    print(f'  SLAKE error: {e}')

# ---- PathVQA ----
print('Processing PathVQA...')
try:
    ds = load_from_disk(f'{DATA_DIR}/pathvqa/dataset')
    for split_name in ['train', 'validation', 'test']:
        target = 'val' if split_name == 'validation' else split_name
        if split_name in ds:
            for i, item in enumerate(ds[split_name]):
                entry = {
                    'id': f'pathvqa_{split_name}_{i}',
                    'dataset': 'PathVQA',
                    'image_path': f'pathvqa/images/{split_name}_{i}.jpg',
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'answer_type': 'OPEN',
                    'question_type': 'pathology',
                }
                unified_data[target].append(entry)
                img_dir = f'{DATA_DIR}/pathvqa/images'
                os.makedirs(img_dir, exist_ok=True)
                if 'image' in item and item['image'] is not None:
                    img_path = f'{img_dir}/{split_name}_{i}.jpg'
                    if not os.path.exists(img_path):
                        item['image'].save(img_path)
    print(f'  PathVQA: processed')
except Exception as e:
    print(f'  PathVQA error: {e}')

# Split val from train if no explicit val set
if not unified_data['val']:
    import random
    random.seed(42)
    random.shuffle(unified_data['train'])
    split_idx = int(len(unified_data['train']) * 0.85)
    unified_data['val'] = unified_data['train'][split_idx:]
    unified_data['train'] = unified_data['train'][:split_idx]

# Save unified dataset files
for split, data in unified_data.items():
    output_path = f'{OUTPUT_DIR}/{split}.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  Saved {split}.json: {len(data)} samples')

print(f'\\nTotal: Train={len(unified_data[\"train\"])}, Val={len(unified_data[\"val\"])}, Test={len(unified_data[\"test\"])}')
print('Dataset preparation complete!')
"

# ============================================================================
# Step 9: Create Directory Structure
# ============================================================================
log_step "Step 9: Creating Project Directories"

mkdir -p "$PROJECT_DIR"/{checkpoints,logs,results,uploads,data/processed,data/knowledge}

log_info "Directory structure created."

# ============================================================================
# Step 10: GPU Optimization Settings
# ============================================================================
log_step "Step 10: Configuring H200 GPU Optimizations"

# Set environment variables for optimal H200 performance
cat > "$PROJECT_DIR/.env" << 'ENV_FILE'
# ============================================================================
# Medical VQA - H200 Environment Variables
# ============================================================================

# CUDA Settings
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=0
TORCH_CUDA_ARCH_LIST="9.0"

# Memory Optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_MEMORY_FRACTION=0.95

# Performance
TOKENIZERS_PARALLELISM=true
OMP_NUM_THREADS=16
MKL_NUM_THREADS=16

# HuggingFace
HF_HOME=/workspace/MedicalVQA/models_cache
TRANSFORMERS_CACHE=/workspace/MedicalVQA/models_cache

# Application
MODEL_PATH=/workspace/MedicalVQA/checkpoints/best_model
DEVICE=cuda
WANDB_PROJECT=medical-vqa-kget

# Flash Attention
FLASH_ATTENTION_FORCE_BUILD=TRUE

# NCCL (for multi-GPU if applicable)
NCCL_DEBUG=INFO
NCCL_IB_DISABLE=0
ENV_FILE

log_info "Environment variables configured in .env"

# ============================================================================
# Step 11: Verify Installation
# ============================================================================
log_step "Step 11: Verifying Installation"

python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU Memory: {gpu_mem:.1f} GB')
    print(f'bf16 Support: {torch.cuda.is_bf16_supported()}')
    print(f'CUDA Arch: {torch.cuda.get_device_capability()}')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import accelerate
print(f'Accelerate: {accelerate.__version__}')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except:
    print('Flash Attention: Not installed')

try:
    import deepspeed
    print(f'DeepSpeed: {deepspeed.__version__}')
except:
    print('DeepSpeed: Not installed')

try:
    import spacy
    nlp = spacy.load('en_core_sci_lg')
    print(f'SciSpacy: loaded en_core_sci_lg')
except:
    print('SciSpacy: Not loaded')

import datasets
print(f'Datasets: {datasets.__version__}')

# Check datasets
import os, json
for split in ['train', 'val', 'test']:
    path = f'data/processed/{split}.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f'  {split}.json: {len(data)} samples')
    else:
        print(f'  {split}.json: NOT FOUND')

print()
print('=' * 60)
print('  SETUP VERIFICATION COMPLETE')
print('=' * 60)
"

# ============================================================================
# Final Summary
# ============================================================================
log_step "Setup Complete!"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Medical VQA - H200 Setup Complete!               ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Next Steps:                                               ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  1. Activate environment:                                  ║${NC}"
echo -e "${GREEN}║     conda activate medvqa                                  ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  2. Start training:                                        ║${NC}"
echo -e "${GREEN}║     python training/train.py \\                             ║${NC}"
echo -e "${GREEN}║       --config config/h200_config.yaml                     ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  3. Start web app:                                         ║${NC}"
echo -e "${GREEN}║     python -m uvicorn webapp.app:app \\                     ║${NC}"
echo -e "${GREEN}║       --host 0.0.0.0 --port 8000                          ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  4. Quick test:                                            ║${NC}"
echo -e "${GREEN}║     python training/train.py --dry_run \\                   ║${NC}"
echo -e "${GREEN}║       --config config/h200_config.yaml                     ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
