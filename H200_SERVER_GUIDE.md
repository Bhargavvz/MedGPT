# ============================================================================
# Medical VQA - Complete H200 Server Guide
# ============================================================================
# GPU: NVIDIA H200 (141GB HBM3e, 4.8 TB/s bandwidth)
# RAM: 1.9 TB System Memory
# ============================================================================

## What You Need to Download

### 1. AI Models (~18 GB total)

| Model | Size | Purpose | HuggingFace ID |
|-------|------|---------|---------------|
| **Qwen2-VL-7B-Instruct** | ~15 GB | Base vision-language model | `Qwen/Qwen2-VL-7B-Instruct` |
| **PubMedBERT** | ~440 MB | Medical knowledge encoder | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| **CLIP ViT-Large** | ~1.7 GB | Vision encoder backbone | `openai/clip-vit-large-patch14` |
| **BioBERT** | ~440 MB | Biomedical text encoding | `dmis-lab/biobert-base-cased-v1.2` |
| **SciSpacy en_core_sci_lg** | ~400 MB | Medical NER | pip install URL |
| **SciSpacy en_ner_bc5cdr_md** | ~100 MB | Drug/disease NER | pip install URL |

### 2. Datasets (~5 GB total)

| Dataset | Images | QA Pairs | Size | HuggingFace ID |
|---------|--------|----------|------|---------------|
| **VQA-RAD** | 315 | 3,515 | ~500 MB | `flaviagiammarino/vqa-rad` |
| **SLAKE** | 642 | 14,028 | ~2 GB | `BoKelvin/SLAKE` |
| **PathVQA** | 4,998 | 32,799 | ~2.5 GB | `flaviagiammarino/path-vqa` |

### 3. System Packages

```
build-essential cmake git git-lfs curl wget unzip htop tmux
libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
libjpeg-dev libpng-dev ninja-build
```

### 4. Python Packages (via pip)

```
torch torchvision torchaudio (CUDA 12.4)
flash-attn xformers triton
transformers peft accelerate bitsandbytes deepspeed
scispacy spacy
fastapi uvicorn
wandb tensorboard gradio
```

---

## Quick Start (3 Commands)

```bash
# 1. One-command setup (downloads everything)
chmod +x scripts/setup_h200_server.sh
bash scripts/setup_h200_server.sh /workspace/MedicalVQA

# 2. Verify setup
python scripts/benchmark_h200.py

# 3. Start training
bash scripts/train_h200.sh
```

---

## Manual Setup (Step by Step)

### Step 1: Environment

```bash
conda create -n medvqa python=3.10 -y
conda activate medvqa

# PyTorch for H200 (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Flash Attention 2
pip install flash-attn --no-build-isolation

# Project dependencies
pip install -r requirements.txt
pip install xformers triton wandb gradio vllm
```

### Step 2: Download Models & Datasets

```bash
python scripts/download_models.py --cache_dir ./models_cache --data_dir ./data/raw
```

### Step 3: Configure Environment

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=./models_cache
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
```

### Step 4: Train

```bash
python training/train.py --config config/h200_config.yaml
```

### Step 5: Launch Web App

```bash
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## H200 Optimization Details

### Why These Settings?

| Setting | Default | H200 Optimized | Why |
|---------|---------|---------------|-----|
| **Quantization** | 4-bit NF4 | Disabled | H200 has 141GB — no need to quantize |
| **Precision** | fp16 | bf16 | H200 Hopper arch has native bf16, better range |
| **Batch Size** | 16 | 32 | 4.7x more VRAM available |
| **Grad Accumulation** | 4 | 2 | Larger batch → fewer accumulation steps |
| **Grad Checkpointing** | Enabled | Disabled | Enough VRAM, disable for 30% speed boost |
| **LoRA Rank** | 64 | 128 | More parameters → better quality |
| **Image Resolution** | 224 | 448 | Higher res → better visual understanding |
| **Data Workers** | 4 | 16 | 1.9TB RAM can handle more prefetch |
| **DeepSpeed ZeRO** | Stage 2 | Stage 1 | Single GPU, Stage 1 is sufficient |
| **Flash Attention** | Off | On | 2-4x faster attention on H200 |

### Expected Performance

| Metric | Estimate |
|--------|----------|
| Training VQA-RAD | ~4 hours |
| Training SLAKE | ~12 hours |
| Training PathVQA | ~24 hours |
| Inference Speed | ~1.2 sec/image |
| GPU Memory Usage | ~85 GB (bf16, no quant) |
| Peak Memory | ~110 GB during training |

---

## Docker Deployment

```bash
# Build H200 image
docker build -f Dockerfile.h200 -t medvqa-h200 .

# Training
docker compose -f docker-compose.h200.yml --profile training up

# Web App (production)
docker compose -f docker-compose.h200.yml up webapp

# With TensorBoard monitoring
docker compose -f docker-compose.h200.yml --profile monitoring up

# Evaluation
docker compose -f docker-compose.h200.yml --profile evaluation up
```

---

## File Structure on Server

```
/workspace/MedicalVQA/
├── config/
│   ├── config.yaml              # Default config
│   ├── h200_config.yaml         # ← H200 optimized config
│   └── model_config.py
├── data/
│   ├── raw/
│   │   ├── vqa_rad/             # ← Downloaded dataset
│   │   ├── slake/               # ← Downloaded dataset
│   │   └── pathvqa/             # ← Downloaded dataset
│   └── processed/
│       ├── train.json           # ← Generated by setup script
│       ├── val.json
│       └── test.json
├── models_cache/                # ← Downloaded AI models (~18GB)
│   ├── models--Qwen--Qwen2-VL-7B-Instruct/
│   ├── models--microsoft--BiomedNLP-PubMedBERT/
│   ├── models--openai--clip-vit-large-patch14/
│   └── models--dmis-lab--biobert-base-cased/
├── checkpoints/                 # ← Training output
├── logs/                        # ← TensorBoard logs
├── scripts/
│   ├── setup_h200_server.sh     # ← One-command setup
│   ├── train_h200.sh            # ← Training launcher
│   ├── download_models.py       # ← Model downloader
│   └── benchmark_h200.py        # ← GPU benchmark
├── training/
│   ├── train.py
│   ├── trainer.py
│   ├── deepspeed_config.json    # Default DeepSpeed
│   └── deepspeed_h200.json      # ← H200 optimized
├── models/
├── evaluation/
├── explainability/
├── inference/
├── webapp/
├── Dockerfile                   # Default
├── Dockerfile.h200              # ← H200 optimized
├── docker-compose.yml           # Default
├── docker-compose.h200.yml      # ← H200 optimized
├── requirements.txt
├── .env                         # ← Generated by setup
└── README.md
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **OOM during training** | Reduce batch_size to 16, enable gradient_checkpointing |
| **Flash Attention fails** | `pip install flash-attn --no-build-isolation` with CUDA dev toolkit |
| **Slow data loading** | Increase `dataloader_num_workers` to 32, increase `prefetch_factor` |
| **NCCL errors** | `export NCCL_IB_DISABLE=1` |
| **Model download fails** | Use `huggingface-cli login` then retry, or use `--resume_download` |
| **SciSpacy not loading** | Manually: `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz` |
| **torch.compile errors** | Set `torch_compile: false` in h200_config.yaml |
