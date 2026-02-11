"""
Prepare Datasets for Training
==============================
Converts downloaded HuggingFace datasets (VQA-RAD, SLAKE, PathVQA)
into the unified JSON format expected by the training pipeline.

Usage:
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --raw_dir ./data/raw --output_dir ./data/processed
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def classify_question(question: str) -> str:
    """Classify question type."""
    q = question.lower().strip()
    if q.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ", "was ", "were ")):
        return "yes_no"
    elif q.startswith(("what", "which")):
        return "what"
    elif q.startswith(("where",)):
        return "where"
    elif q.startswith(("how many", "how much")):
        return "how_many"
    return "other"

def classify_answer_type(answer: str) -> str:
    """Classify if answer is closed (yes/no) or open."""
    a = answer.lower().strip()
    if a in ("yes", "no", "true", "false"):
        return "closed"
    return "open"

def load_vqa_rad(raw_dir: str) -> list:
    """Load VQA-RAD dataset from HuggingFace cached format."""
    ds_dir = Path(raw_dir) / "vqa_rad"
    samples = []

    try:
        from datasets import load_from_disk, load_dataset

        # Try loading from saved disk format
        dataset_path = ds_dir / "dataset"
        if dataset_path.exists():
            ds = load_from_disk(str(dataset_path))
        else:
            # Fall back to re-downloading
            ds = load_dataset("flaviagiammarino/vqa-rad", cache_dir=str(ds_dir))

        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in ds.items():
            for idx, item in enumerate(split_data):
                # Save image if present
                img_path = ""
                if "image" in item and item["image"] is not None:
                    img_filename = f"vqa_rad_{split_name}_{idx}.jpg"
                    img_full_path = img_dir / img_filename
                    if not img_full_path.exists():
                        try:
                            item["image"].save(str(img_full_path), "JPEG")
                        except Exception:
                            pass
                    img_path = str(img_full_path)

                question = item.get("question", "")
                answer = str(item.get("answer", ""))

                sample = {
                    "id": f"vqa_rad_{split_name}_{idx}",
                    "image": img_path,
                    "question": question,
                    "answer": answer,
                    "modality": item.get("modality", "Other") if "modality" in item else "Other",
                    "organ": item.get("organ", "") if "organ" in item else "",
                    "disease": "",
                    "question_type": classify_question(question),
                    "answer_type": classify_answer_type(answer),
                    "knowledge_snippet": "",
                    "source_dataset": "VQA-RAD",
                    "split": split_name if split_name in ("train", "test") else "train",
                }
                samples.append(sample)

        print(f"  ✓ VQA-RAD: {len(samples)} samples loaded")

    except Exception as e:
        print(f"  ✗ VQA-RAD error: {e}")
        # Try loading from JSON files if they exist
        for json_file in ds_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        samples.append({
                            "id": f"vqa_rad_{idx}",
                            "image": item.get("image_name", item.get("image", "")),
                            "question": item.get("question", ""),
                            "answer": str(item.get("answer", "")),
                            "modality": item.get("modality", "Other"),
                            "organ": item.get("organ", ""),
                            "disease": "",
                            "question_type": classify_question(item.get("question", "")),
                            "answer_type": classify_answer_type(str(item.get("answer", ""))),
                            "knowledge_snippet": "",
                            "source_dataset": "VQA-RAD",
                            "split": "train",
                        })
                print(f"  ✓ VQA-RAD (JSON): {len(samples)} samples loaded")
            except Exception:
                pass

    return samples

def load_slake(raw_dir: str) -> list:
    """Load SLAKE dataset from HuggingFace cached format."""
    ds_dir = Path(raw_dir) / "slake"
    samples = []

    try:
        from datasets import load_from_disk, load_dataset

        dataset_path = ds_dir / "dataset"
        if dataset_path.exists():
            ds = load_from_disk(str(dataset_path))
        else:
            ds = load_dataset("BoKelvin/SLAKE", cache_dir=str(ds_dir))

        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in ds.items():
            for idx, item in enumerate(split_data):
                # Only English questions
                lang = item.get("q_lang", "en")
                if lang != "en" and "q_lang" in item:
                    continue

                img_path = ""
                if "image" in item and item["image"] is not None:
                    img_filename = f"slake_{split_name}_{idx}.jpg"
                    img_full_path = img_dir / img_filename
                    if not img_full_path.exists():
                        try:
                            item["image"].save(str(img_full_path), "JPEG")
                        except Exception:
                            pass
                    img_path = str(img_full_path)

                question = item.get("question", "")
                answer = str(item.get("answer", ""))

                sample = {
                    "id": f"slake_{split_name}_{idx}",
                    "image": img_path,
                    "question": question,
                    "answer": answer,
                    "modality": item.get("modality", item.get("img_type", "Other")),
                    "organ": item.get("organ", item.get("location", "")),
                    "disease": "",
                    "question_type": classify_question(question),
                    "answer_type": item.get("answer_type", classify_answer_type(answer)),
                    "knowledge_snippet": item.get("knowledge", ""),
                    "source_dataset": "SLAKE",
                    "split": split_name if split_name in ("train", "validation", "test") else "train",
                }
                # Normalize split names
                if sample["split"] == "validation":
                    sample["split"] = "val"
                samples.append(sample)

        print(f"  ✓ SLAKE: {len(samples)} samples loaded")

    except Exception as e:
        print(f"  ✗ SLAKE error: {e}")

    return samples

def load_pathvqa(raw_dir: str) -> list:
    """Load PathVQA dataset from HuggingFace cached format."""
    ds_dir = Path(raw_dir) / "pathvqa"
    samples = []

    try:
        from datasets import load_from_disk, load_dataset

        dataset_path = ds_dir / "dataset"
        if dataset_path.exists():
            ds = load_from_disk(str(dataset_path))
        else:
            ds = load_dataset("flaviagiammarino/path-vqa", cache_dir=str(ds_dir))

        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in ds.items():
            for idx, item in enumerate(split_data):
                img_path = ""
                if "image" in item and item["image"] is not None:
                    img_filename = f"pathvqa_{split_name}_{idx}.jpg"
                    img_full_path = img_dir / img_filename
                    if not img_full_path.exists():
                        try:
                            item["image"].save(str(img_full_path), "JPEG")
                        except Exception:
                            pass
                    img_path = str(img_full_path)

                question = item.get("question", "")
                answer = str(item.get("answer", ""))

                sample = {
                    "id": f"pathvqa_{split_name}_{idx}",
                    "image": img_path,
                    "question": question,
                    "answer": answer,
                    "modality": "Pathology",
                    "organ": "",
                    "disease": "",
                    "question_type": classify_question(question),
                    "answer_type": classify_answer_type(answer),
                    "knowledge_snippet": "",
                    "source_dataset": "PathVQA",
                    "split": split_name if split_name in ("train", "validation", "test") else "train",
                }
                if sample["split"] == "validation":
                    sample["split"] = "val"
                samples.append(sample)

        print(f"  ✓ PathVQA: {len(samples)} samples loaded")

    except Exception as e:
        print(f"  ✗ PathVQA error: {e}")

    return samples


def split_data(samples: list, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15) -> dict:
    """Split samples into train/val/test sets.
    
    Uses existing split info if available, otherwise splits randomly.
    """
    # Check if samples already have meaningful split assignments
    splits = defaultdict(list)
    for s in samples:
        split = s.get("split", "train")
        if split in ("train", "val", "test"):
            splits[split].append(s)
        else:
            splits["train"].append(s)

    # If we have all three splits, use them
    if splits["train"] and splits["test"]:
        if not splits["val"]:
            # Split some training data for validation
            random.shuffle(splits["train"])
            n_val = max(1, int(len(splits["train"]) * 0.15))
            splits["val"] = splits["train"][:n_val]
            splits["train"] = splits["train"][n_val:]
        return dict(splits)

    # Otherwise, random split
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--raw_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Medical VQA - Dataset Preparation")
    print("=" * 60)
    print(f"  Raw dir:    {os.path.abspath(args.raw_dir)}")
    print(f"  Output dir: {os.path.abspath(args.output_dir)}")
    print("=" * 60)
    print()

    # Load all datasets
    print("Loading datasets...")
    all_samples = []

    vqa_rad = load_vqa_rad(args.raw_dir)
    all_samples.extend(vqa_rad)

    slake = load_slake(args.raw_dir)
    all_samples.extend(slake)

    pathvqa = load_pathvqa(args.raw_dir)
    all_samples.extend(pathvqa)

    print(f"\n  Total samples loaded: {len(all_samples)}")

    if not all_samples:
        print("\n  ✗ No samples found! Check your data/raw/ directory.")
        return

    # Filter out samples without images
    valid = [s for s in all_samples if s["image"] and os.path.exists(s["image"])]
    print(f"  Samples with valid images: {len(valid)}")
    
    if not valid:
        print("  ⚠ No valid images found. Using all samples (images may need downloading)")
        valid = all_samples

    # Split into train/val/test
    print("\nSplitting data...")
    splits = split_data(valid)

    # Update split field
    for split_name, split_data_list in splits.items():
        for s in split_data_list:
            s["split"] = split_name

    # Save
    print("\nSaving...")
    stats = {}
    for split_name, split_data_list in splits.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(output_path, "w") as f:
            json.dump(split_data_list, f, indent=2)
        stats[split_name] = len(split_data_list)
        print(f"  ✓ {split_name}.json: {len(split_data_list)} samples")

    # Print summary
    print()
    print("=" * 60)
    print("  Dataset Summary")
    print("=" * 60)

    # Source distribution
    source_counts = defaultdict(int)
    for s in valid:
        source_counts[s["source_dataset"]] += 1
    print("\n  By source:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src:12s}: {count:6d}")

    # Question type distribution
    qtype_counts = defaultdict(int)
    for s in valid:
        qtype_counts[s["question_type"]] += 1
    print("\n  By question type:")
    for qt, count in sorted(qtype_counts.items()):
        print(f"    {qt:12s}: {count:6d}")

    # Answer type distribution
    atype_counts = defaultdict(int)
    for s in valid:
        atype_counts[s["answer_type"]] += 1
    print("\n  By answer type:")
    for at, count in sorted(atype_counts.items()):
        print(f"    {at:12s}: {count:6d}")

    print(f"\n  Total: {sum(stats.values())} samples")
    print(f"  Train: {stats.get('train', 0)} | Val: {stats.get('val', 0)} | Test: {stats.get('test', 0)}")
    print("=" * 60)
    print("  Done! Ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
