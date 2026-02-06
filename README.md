# Knowledge-Guided Explainable Transformer for Medical Visual Question Answering

<div align="center">

![Medical VQA](https://img.shields.io/badge/Medical-VQA-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered system for answering questions about medical images with explainable reasoning**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Training](#training) • [API](#api) • [Evaluation](#evaluation)

</div>

---

## 🌟 Features

- **🧠 Qwen2-VL-7B Base Model** - State-of-the-art vision-language understanding
- **📚 Medical Knowledge Integration** - BioBERT/PubMedBERT for domain-specific knowledge
- **🔬 Explainable AI** - Grad-CAM, Attention Rollout, Integrated Gradients
- **⚡ Efficient Fine-tuning** - LoRA/QLoRA with 4-bit quantization
- **🌐 Web Interface** - FastAPI backend with modern HTML/CSS/JS frontend
- **🐳 Docker Ready** - Easy deployment with GPU support

---

## 📁 Project Structure

```
medical-vqa/
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   └── model_config.py        # Python dataclass configs
├── data/                       # Data handling
│   ├── schema.json            # Dataset schema
│   └── dataset_loader.py      # Data loading utilities
├── preprocess/                 # Preprocessing modules
│   ├── dicom_processor.py     # DICOM to PNG conversion
│   ├── image_augmentation.py  # Medical image augmentation
│   ├── text_processor.py      # Text preprocessing
│   └── knowledge_retriever.py # UMLS/SciSpacy integration
├── models/                     # Model architecture
│   ├── vision_encoder.py      # Vision encoder (CLIP ViT)
│   ├── knowledge_encoder.py   # Knowledge encoder (BioBERT)
│   ├── fusion_module.py       # Cross-attention fusion
│   ├── explanation_head.py    # Rationale generation
│   └── medical_vqa_model.py   # Main VQA model
├── training/                   # Training pipeline
│   ├── loss_functions.py      # Multi-objective losses
│   ├── trainer.py             # Custom trainer
│   └── train.py               # Training script
├── evaluation/                 # Evaluation
│   ├── metrics.py             # VQA metrics
│   └── evaluate.py            # Evaluation pipeline
├── explainability/             # XAI modules
│   ├── grad_cam.py            # Grad-CAM implementations
│   ├── attention_vis.py       # Attention visualization
│   └── integrated_gradients.py # Integrated gradients
├── inference/                  # Inference pipeline
│   └── pipeline.py            # End-to-end inference
├── webapp/                     # Web application
│   ├── app.py                 # FastAPI backend
│   └── static/                # Frontend files
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 24GB+ VRAM (recommended)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/medical-vqa.git
cd medical-vqa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build image
docker build -t medical-vqa .

# Run with GPU
docker-compose up -d
```

---

## 📊 Datasets

### Supported Datasets

| Dataset | Modality | Samples | Task |
|---------|----------|---------|------|
| VQA-RAD | Multi | 3,515 | VQA |
| SLAKE | Multi | 14,028 | VQA |
| PathVQA | Pathology | 32,799 | VQA |
| MedVQA | Multi | 4,706 | VQA |

### Dataset Format

```json
{
  "image": "path/to/image.png",
  "question": "What type of imaging is this?",
  "answer": "chest x-ray",
  "modality": "xray",
  "organ": "lung",
  "disease": "normal",
  "knowledge_snippet": "Chest X-ray is a radiological examination..."
}
```

### Data Preparation

```bash
# Convert DICOM to PNG
python preprocess/dicom_processor.py \
    --input_dir data/raw/dicom \
    --output_dir data/processed/images

# Prepare unified dataset
python scripts/prepare_dataset.py \
    --vqa_rad_path data/raw/vqa-rad \
    --slake_path data/raw/slake \
    --output_path data/processed/unified_vqa.json
```

---

## 🏋️ Training

### Basic Training

```bash
python training/train.py \
    --config config/config.yaml \
    --output_dir ./checkpoints \
    --num_epochs 15 \
    --batch_size 16
```

### Training with DeepSpeed

```bash
deepspeed --num_gpus=1 training/train.py \
    --deepspeed training/deepspeed_config.json \
    --config config/config.yaml
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_epochs` | 15 | Number of training epochs |
| `--batch_size` | 16 | Training batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--lora_r` | 64 | LoRA rank |
| `--freeze_vision_epochs` | 3 | Epochs to freeze vision encoder |

---

## 🔮 Inference

### Python API

```python
from inference import VQAInference

# Initialize pipeline
pipeline = VQAInference(model_path="./checkpoints/best_model")

# Single prediction
result = pipeline.predict(
    image="xray.png",
    question="What abnormalities are visible?",
    generate_explanation=True,
    generate_heatmap=True
)

print(f"Answer: {result['answer']}")
print(f"Explanation: {result['explanation']}")
```

### REST API

```bash
# Start server
uvicorn webapp.app:app --host 0.0.0.0 --port 8000

# Query endpoint
curl -X POST "http://localhost:8000/api/vqa" \
    -F "image=@xray.png" \
    -F "question=What is the diagnosis?"
```

---

## 🌐 Web Application

### Running the Web App

```bash
# Start server
python webapp/app.py

# Access at http://localhost:8000
```

### Features

- 📤 Drag-and-drop image upload
- ❓ Natural language questions
- 💡 Explainable answers with rationale
- 🔥 Attention heatmap visualization
- 📥 Downloadable reports
- 🌙 Dark mode support

---

## 📈 Evaluation

### Run Evaluation

```bash
python evaluation/evaluate.py \
    --model_path ./checkpoints/best_model \
    --test_file data/test.json \
    --output_dir ./results
```

### Metrics

| Metric | VQA-RAD | SLAKE |
|--------|---------|-------|
| Accuracy | 75%+ | 72%+ |
| BLEU-1 | 0.68 | 0.65 |
| ROUGE-L | 0.71 | 0.68 |

### Ablation Studies

```bash
python evaluation/evaluate.py --run_ablation
```

---

## 🔍 Explainability

### Generate Explanations

```python
from explainability import GradCAM, AttentionRollout

# Grad-CAM
grad_cam = GradCAM(model.vision_encoder)
heatmap = grad_cam(image_tensor)

# Attention Rollout
rollout = AttentionRollout(model)
attention = rollout(image_tensor)
```

### Visualization

The system provides:
- **Grad-CAM heatmaps** - Highlight important image regions
- **Attention maps** - Token-level attention patterns
- **Textual rationales** - Step-by-step reasoning explanations

---

## ⚙️ Configuration

### Model Configuration

```yaml
# config/config.yaml
model:
  base_model: "Qwen/Qwen2-VL-7B-Instruct"
  vision_encoder: "openai/clip-vit-large-patch14"
  knowledge_encoder: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

lora:
  enabled: true
  r: 64
  lora_alpha: 128

training:
  num_epochs: 15
  batch_size: 16
  learning_rate: 2e-5
```

---

## 🐳 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Cloud Deployment

The application is ready for deployment on:
- AWS EC2 (with GPU instances)
- Google Cloud Compute Engine
- Azure Virtual Machines

---

## ⚠️ Disclaimer

**This system is for research and educational purposes only.** It should NOT be used for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - Base vision-language model
- [BioBERT](https://github.com/dmis-lab/biobert) - Biomedical language model
- [VQA-RAD](https://www.nature.com/articles/sdata2018251) - VQA dataset
- [SLAKE](https://github.com/Sadayuki-Sato/SLAKE) - Semantic VQA dataset

---

<div align="center">

**Built with ❤️ for advancing medical AI**

</div>
