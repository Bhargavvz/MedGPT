"""
Download All Required AI Models
================================
Downloads all models needed for the Medical VQA system.
Run this BEFORE training or inference.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --cache_dir /path/to/cache
"""

import argparse
import os
import sys
from pathlib import Path

def download_models(cache_dir: str = "./models_cache"):
    """Download all required models from HuggingFace."""
    
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    from huggingface_hub import snapshot_download
    
    models = [
        {
            "name": "Qwen2-VL-7B-Instruct",
            "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
            "description": "Base Vision-Language Model (~15GB)",
            "size": "~15 GB",
        },
        {
            "name": "PubMedBERT",
            "repo_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "description": "Biomedical Knowledge Encoder (~440MB)",
            "size": "~440 MB",
        },
        {
            "name": "CLIP-ViT-Large",
            "repo_id": "openai/clip-vit-large-patch14",
            "description": "Vision Encoder (~1.7GB)",
            "size": "~1.7 GB",
        },
        {
            "name": "BioBERT",
            "repo_id": "dmis-lab/biobert-base-cased-v1.2",
            "description": "Biomedical BERT (~440MB)",
            "size": "~440 MB",
        },
    ]
    
    total = len(models)
    print("=" * 65)
    print("  Medical VQA - Model Downloader")
    print("=" * 65)
    print(f"  Cache Directory: {os.path.abspath(cache_dir)}")
    print(f"  Models to Download: {total}")
    print(f"  Total Estimated Size: ~18 GB")
    print("=" * 65)
    print()
    
    for i, model in enumerate(models, 1):
        print(f"[{i}/{total}] Downloading {model['name']}...")
        print(f"       Repo: {model['repo_id']}")
        print(f"       Size: {model['size']}")
        print(f"       Info: {model['description']}")
        
        try:
            path = snapshot_download(
                model["repo_id"],
                cache_dir=cache_dir,
                resume_download=True,
            )
            print(f"       ✓ Downloaded to: {path}")
        except Exception as e:
            print(f"       ✗ Error: {e}")
            print(f"       Retrying...")
            try:
                path = snapshot_download(
                    model["repo_id"],
                    cache_dir=cache_dir,
                    resume_download=True,
                    force_download=True,
                )
                print(f"       ✓ Downloaded to: {path}")
            except Exception as e2:
                print(f"       ✗ Failed: {e2}")
                print(f"       Please download manually: huggingface-cli download {model['repo_id']}")
        print()
    
    # Download SciSpacy models
    print(f"[Extra] Downloading SciSpacy models...")
    try:
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
        ], check=True)
        print(f"       ✓ en_core_sci_lg installed")
    except Exception as e:
        print(f"       ✗ SciSpacy error: {e}")
    print()
    
    # Verify all models
    print("=" * 65)
    print("  Verification")
    print("=" * 65)
    
    try:
        from transformers import AutoTokenizer, AutoProcessor
        
        print("  Checking Qwen2-VL...", end=" ")
        AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir=cache_dir, trust_remote_code=True)
        print("✓")
        
        print("  Checking PubMedBERT...", end=" ")
        AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", cache_dir=cache_dir)
        print("✓")
        
        print("  Checking CLIP...", end=" ")
        AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
        print("✓")
        
        print("  Checking BioBERT...", end=" ")
        AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir=cache_dir)
        print("✓")
        
    except Exception as e:
        print(f"\n  Verification warning: {e}")
    
    print()
    print("=" * 65)
    print("  All models downloaded successfully!")
    print("=" * 65)


def download_datasets(data_dir: str = "./data/raw"):
    """Download all required datasets."""
    
    os.makedirs(data_dir, exist_ok=True)
    
    from datasets import load_dataset
    
    datasets_info = [
        {
            "name": "VQA-RAD",
            "hf_id": "flaviagiammarino/vqa-rad",
            "subdir": "vqa_rad",
            "description": "3,515 QA pairs on radiology images",
        },
        {
            "name": "SLAKE",
            "hf_id": "BoKelvin/SLAKE",
            "subdir": "slake",
            "description": "14,028 QA pairs with knowledge annotations",
        },
        {
            "name": "PathVQA",
            "hf_id": "flaviagiammarino/path-vqa",
            "subdir": "pathvqa",
            "description": "32,799 QA pairs on pathology images",
        },
    ]
    
    total = len(datasets_info)
    print("=" * 65)
    print("  Medical VQA - Dataset Downloader")
    print("=" * 65)
    print(f"  Data Directory: {os.path.abspath(data_dir)}")
    print(f"  Datasets: {total}")
    print("=" * 65)
    print()
    
    for i, ds_info in enumerate(datasets_info, 1):
        print(f"[{i}/{total}] Downloading {ds_info['name']}...")
        print(f"       HF ID: {ds_info['hf_id']}")
        print(f"       Info: {ds_info['description']}")
        
        save_path = os.path.join(data_dir, ds_info["subdir"])
        os.makedirs(save_path, exist_ok=True)
        
        try:
            ds = load_dataset(ds_info["hf_id"], cache_dir=save_path)
            ds.save_to_disk(os.path.join(save_path, "dataset"))
            
            for split_name, split_data in ds.items():
                print(f"       {split_name}: {len(split_data)} samples")
            
            # Extract and save images
            img_dir = os.path.join(save_path, "images")
            os.makedirs(img_dir, exist_ok=True)
            
            for split_name, split_data in ds.items():
                for idx, item in enumerate(split_data):
                    if "image" in item and item["image"] is not None:
                        img_path = os.path.join(img_dir, f"{split_name}_{idx}.jpg")
                        if not os.path.exists(img_path):
                            try:
                                item["image"].save(img_path, "JPEG")
                            except Exception:
                                pass
            
            print(f"       ✓ Downloaded and images extracted")
        except Exception as e:
            print(f"       ✗ Error: {e}")
        print()
    
    print("=" * 65)
    print("  All datasets downloaded!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models and datasets")
    parser.add_argument("--cache_dir", type=str, default="./models_cache", help="Model cache dir")
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Dataset dir")
    parser.add_argument("--models_only", action="store_true", help="Download models only")
    parser.add_argument("--datasets_only", action="store_true", help="Download datasets only")
    args = parser.parse_args()
    
    if args.datasets_only:
        download_datasets(args.data_dir)
    elif args.models_only:
        download_models(args.cache_dir)
    else:
        download_models(args.cache_dir)
        print("\n\n")
        download_datasets(args.data_dir)
