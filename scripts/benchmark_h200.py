"""
H200 GPU Benchmark & Verification Script
==========================================
Validates H200 GPU performance and project readiness.

Usage:
    python scripts/benchmark_h200.py
"""

import os
import sys
import time
import json
from pathlib import Path

def check_gpu():
    """Check GPU availability and specs."""
    import torch

    print("=" * 65)
    print("  H200 GPU Benchmark & System Verification")
    print("=" * 65)
    print()

    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    print("1. GPU Information")
    print("-" * 40)
    print(f"   Name:           {props.name}")
    print(f"   Compute Cap:    {props.major}.{props.minor}")
    print(f"   Total Memory:   {props.total_memory / 1e9:.1f} GB")
    print(f"   SMs:            {props.multi_processor_count}")
    print(f"   CUDA Version:   {torch.version.cuda}")
    print(f"   PyTorch:        {torch.__version__}")
    print(f"   bf16 Support:   {torch.cuda.is_bf16_supported()}")
    print()

    return True


def benchmark_memory():
    """Benchmark memory allocation and bandwidth."""
    import torch

    print("2. Memory Benchmark")
    print("-" * 40)

    device = torch.device("cuda:0")

    # Test large tensor allocation (simulate 7B model in bf16)
    sizes_gb = [1, 5, 10, 20, 50]
    for size_gb in sizes_gb:
        try:
            numel = int(size_gb * 1e9 / 2)  # bf16 = 2 bytes
            start = time.time()
            t = torch.empty(numel, dtype=torch.bfloat16, device=device)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"   Alloc {size_gb:4d} GB bf16: {elapsed*1000:.1f} ms ✓")
            del t
            torch.cuda.empty_cache()
        except RuntimeError:
            print(f"   Alloc {size_gb:4d} GB bf16: OOM ✗")
            break

    # Memory bandwidth test
    size = 1024 * 1024 * 256  # 512 MB in bf16
    a = torch.randn(size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(5):
        c = a + b
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        c = a + b
    torch.cuda.synchronize()
    elapsed = time.time() - start

    bytes_moved = 3 * size * 2 * n_iters  # read a, b; write c
    bandwidth_tbs = bytes_moved / elapsed / 1e12
    print(f"   Memory Bandwidth: {bandwidth_tbs:.2f} TB/s (H200 peak: 4.8 TB/s)")

    del a, b, c
    torch.cuda.empty_cache()
    print()


def benchmark_compute():
    """Benchmark compute performance."""
    import torch

    print("3. Compute Benchmark")
    print("-" * 40)

    device = torch.device("cuda:0")

    # Matrix multiply benchmark (simulates transformer layers)
    sizes = [(4096, 4096), (4096, 11008), (8192, 8192)]

    for m, n in sizes:
        a = torch.randn(m, n, dtype=torch.bfloat16, device=device)
        b = torch.randn(n, m, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            c = torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 50
        start = time.time()
        for _ in range(n_iters):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        flops = 2 * m * n * m * n_iters
        tflops = flops / elapsed / 1e12
        print(f"   MatMul [{m}x{n}] x [{n}x{m}]: {tflops:.1f} TFLOPS (bf16)")

        del a, b, c
        torch.cuda.empty_cache()

    print()


def benchmark_flash_attention():
    """Benchmark Flash Attention 2."""
    import torch

    print("4. Flash Attention Benchmark")
    print("-" * 40)

    try:
        from flash_attn import flash_attn_func

        device = torch.device("cuda:0")
        batch, heads, seq_len, head_dim = 4, 32, 2048, 128

        q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            out = flash_attn_func(q, k, v)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 50
        start = time.time()
        for _ in range(n_iters):
            out = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        ms_per_iter = elapsed / n_iters * 1000
        print(f"   Flash Attn (B={batch}, H={heads}, S={seq_len}): {ms_per_iter:.2f} ms/iter ✓")

        del q, k, v, out
        torch.cuda.empty_cache()

    except ImportError:
        print("   Flash Attention 2 not installed ✗")
        print("   Install: pip install flash-attn --no-build-isolation")
    print()


def check_dependencies():
    """Check all required packages."""
    print("5. Dependency Check")
    print("-" * 40)

    packages = [
        ("torch",           "PyTorch"),
        ("transformers",    "Transformers"),
        ("peft",            "PEFT (LoRA)"),
        ("accelerate",      "Accelerate"),
        ("deepspeed",       "DeepSpeed"),
        ("bitsandbytes",    "BitsAndBytes"),
        ("datasets",        "HF Datasets"),
        ("flash_attn",      "Flash Attention"),
        ("spacy",           "SpaCy"),
        ("fastapi",         "FastAPI"),
        ("uvicorn",         "Uvicorn"),
        ("wandb",           "W&B"),
        ("captum",          "Captum"),
        ("grad_cam",        "Grad-CAM"),
        ("cv2",             "OpenCV"),
        ("PIL",             "Pillow"),
        ("albumentations",  "Albumentations"),
        ("nltk",            "NLTK"),
        ("rouge_score",     "ROUGE"),
        ("sklearn",         "Scikit-learn"),
    ]

    installed = 0
    for pkg, name in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "OK")
            print(f"   ✓ {name:20s} {ver}")
            installed += 1
        except ImportError:
            print(f"   ✗ {name:20s} NOT INSTALLED")
        except Exception as e:
            print(f"   ⚠ {name:20s} IMPORT ERROR ({type(e).__name__})")

    print(f"\n   {installed}/{len(packages)} packages installed")
    print()


def check_datasets():
    """Check downloaded datasets."""
    print("6. Dataset Check")
    print("-" * 40)

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # Check processed
    for split in ["train", "val", "test"]:
        path = data_dir / "processed" / f"{split}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"   ✓ {split}.json: {len(data)} samples")
        else:
            print(f"   ✗ {split}.json: NOT FOUND")

    # Check raw
    for ds_name in ["vqa_rad", "slake", "pathvqa"]:
        ds_path = data_dir / "raw" / ds_name
        if ds_path.exists():
            n_files = sum(1 for _ in ds_path.rglob("*") if _.is_file())
            print(f"   ✓ {ds_name}: {n_files} files")
        else:
            print(f"   ✗ {ds_name}: NOT FOUND")

    print()


def check_models():
    """Check downloaded models."""
    print("7. Model Check")
    print("-" * 40)

    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "models_cache"

    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        print(f"   Cache directory: {cache_dir}")
        print(f"   Total size: {total_size / 1e9:.1f} GB")
    else:
        print(f"   ✗ Cache not found at {cache_dir}")
        print(f"   Run: python scripts/download_models.py")

    # Try loading
    models = {
        "Qwen2-VL": "Qwen/Qwen2-VL-7B-Instruct",
        "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "CLIP": "openai/clip-vit-large-patch14",
    }

    for name, repo_id in models.items():
        try:
            from huggingface_hub import model_info
            info = model_info(repo_id)
            print(f"   ✓ {name}: available on HuggingFace")
        except Exception:
            print(f"   ? {name}: could not verify")

    print()


def estimate_training_time():
    """Estimate training time on H200."""
    import torch

    print("8. Training Time Estimate")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("   Cannot estimate without GPU")
        return

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9

    # Estimates based on H200 benchmarks
    datasets = {
        "VQA-RAD":  {"train": 3064,  "per_sample_ms": 85},
        "SLAKE":    {"train": 9849,  "per_sample_ms": 85},
        "PathVQA":  {"train": 19654, "per_sample_ms": 90},
    }

    epochs = 15
    print(f"   GPU Memory: {vram_gb:.0f} GB")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: 32 (H200 optimized)")
    print()

    total_hours = 0
    for ds_name, info in datasets.items():
        steps = info["train"] * epochs / 32  # batch_size 32
        hours = steps * info["per_sample_ms"] * 32 / 1000 / 3600
        total_hours += hours
        print(f"   {ds_name:10s}: ~{hours:.1f} hours ({info['train']} samples)")

    print(f"\n   Total Estimated: ~{total_hours:.1f} hours")
    print(f"   (Combined training on all 3 datasets)")
    print()


def main():
    ok = check_gpu()
    if ok:
        benchmark_memory()
        benchmark_compute()
        benchmark_flash_attention()
    check_dependencies()
    check_datasets()
    check_models()
    if ok:
        estimate_training_time()

    print("=" * 65)
    print("  Benchmark Complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
