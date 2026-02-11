"""
Dataset Loader Module
=====================
Unified dataset loading for medical VQA datasets.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from loguru import logger

from preprocess.image_augmentation import MedicalImageAugmentation, get_augmentation_by_modality
from preprocess.text_processor import TextProcessor
from preprocess.knowledge_retriever import KnowledgeRetriever


@dataclass
class VQASample:
    """Represents a single VQA sample."""
    id: str
    image_path: str
    question: str
    answer: str
    modality: str = "Unknown"
    organ: str = ""
    disease: str = ""
    question_type: str = "other"
    answer_type: str = "open"
    knowledge_snippet: str = ""
    source_dataset: str = ""
    split: str = "train"


class MedicalVQADataset(Dataset):
    """
    Unified Medical VQA Dataset.
    
    Supports VQA-RAD, SLAKE, PathVQA, and custom datasets.
    """
    
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        processor: Optional[AutoProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        augmentation: Optional[MedicalImageAugmentation] = None,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        max_question_length: int = 128,
        max_answer_length: int = 256,
        image_size: int = 224,
        split: str = "train",
        include_knowledge: bool = True,
        return_raw: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSON data file
            image_dir: Directory containing images
            processor: HuggingFace processor for Qwen-VL
            text_processor: Text preprocessing instance
            augmentation: Image augmentation pipeline
            knowledge_retriever: Knowledge retrieval instance
            max_question_length: Maximum question token length
            max_answer_length: Maximum answer token length
            image_size: Target image size
            split: Data split (train/val/test)
            include_knowledge: Whether to include knowledge snippets
            return_raw: Return raw data instead of processed tensors
        """
        self.image_dir = Path(image_dir)
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.image_size = image_size
        self.split = split
        self.include_knowledge = include_knowledge
        self.return_raw = return_raw
        
        # Load data
        self.samples = self._load_data(data_file)
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Initialize processor
        self.processor = processor
        
        # Initialize augmentation
        self.augmentation = augmentation or MedicalImageAugmentation(
            image_size=image_size,
            is_training=(split == "train")
        )
        
        # Initialize text processor
        self.text_processor = text_processor or TextProcessor(
            max_question_length=max_question_length,
            max_answer_length=max_answer_length
        )
        
        # Initialize knowledge retriever
        self.knowledge_retriever = knowledge_retriever
    
    def _load_data(self, data_file: str) -> List[VQASample]:
        """Load data from JSON file."""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            # Handle different JSON formats
            if isinstance(item, dict):
                samples.append(VQASample(
                    id=item.get('id', str(len(samples))),
                    image_path=item.get('image', item.get('image_path', '')),
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    modality=item.get('modality', 'Unknown'),
                    organ=item.get('organ', ''),
                    disease=item.get('disease', ''),
                    question_type=item.get('question_type', 'other'),
                    answer_type=item.get('answer_type', 'open'),
                    knowledge_snippet=item.get('knowledge_snippet', ''),
                    source_dataset=item.get('source_dataset', ''),
                    split=item.get('split', self.split)
                ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image - handle absolute paths and relative paths
        img_path = Path(sample.image_path) if sample.image_path else None
        if img_path and img_path.is_absolute() and img_path.is_file():
            image_path = img_path
        elif img_path and sample.image_path and img_path.is_file():
            image_path = img_path
        else:
            image_path = self.image_dir / sample.image_path if sample.image_path else self.image_dir / "missing.jpg"
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return blank image
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Apply augmentation
        augmented = self.augmentation(image)
        image_tensor = augmented['image']
        
        # Process question
        question = self.text_processor.process_question(sample.question)
        
        # Process answer
        answer = self.text_processor.process_answer(sample.answer)
        
        # Get knowledge snippet
        knowledge_snippet = sample.knowledge_snippet
        if self.include_knowledge and not knowledge_snippet and self.knowledge_retriever:
            knowledge_data = self.knowledge_retriever.retrieve_for_vqa(
                question,
                modality=sample.modality
            )
            knowledge_snippet = knowledge_data.get('knowledge_snippet', '')
        
        if self.return_raw:
            return {
                'id': sample.id,
                'image': image_tensor,
                'question': question,
                'answer': answer,
                'modality': sample.modality,
                'knowledge_snippet': knowledge_snippet,
                'question_type': sample.question_type,
                'source_dataset': sample.source_dataset,
            }
        
        # Format input for model
        if self.processor:
            # Use HuggingFace processor for Qwen-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.fromarray(
                            (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            if isinstance(image_tensor, torch.Tensor) else image_tensor
                        )},
                        {"type": "text", "text": self._format_prompt(question, knowledge_snippet)}
                    ]
                }
            ]
            
            # Process with Qwen processor
            inputs = self.processor(
                text=messages,
                images=None,  # Already included in messages
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_question_length + self.max_answer_length
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': image_tensor,
                'labels': self._encode_answer(answer),
                'id': sample.id,
                'question_type': sample.question_type,
            }
        else:
            # Return basic format
            return {
                'id': sample.id,
                'image': image_tensor,
                'question': question,
                'answer': answer,
                'knowledge_snippet': knowledge_snippet,
                'modality': sample.modality,
                'question_type': sample.question_type,
            }
    
    def _format_prompt(self, question: str, knowledge: str = "") -> str:
        """Format the prompt with knowledge context."""
        prompt = question
        
        if knowledge:
            prompt = f"Medical Context: {knowledge}\n\nQuestion: {question}"
        
        return prompt
    
    def _encode_answer(self, answer: str) -> torch.Tensor:
        """Encode answer to token IDs."""
        if self.text_processor:
            encoded = self.text_processor.tokenize_answer(answer, return_tensors="pt")
            return encoded['input_ids'].squeeze(0)
        return torch.tensor([])


class VQARADDataset(MedicalVQADataset):
    """VQA-RAD specific dataset loader."""
    
    @classmethod
    def from_vqa_rad(
        cls,
        data_dir: str,
        split: str = "train",
        **kwargs
    ) -> "VQARADDataset":
        """
        Load VQA-RAD dataset.
        
        Args:
            data_dir: VQA-RAD data directory
            split: train/val/test
            **kwargs: Additional arguments for base class
            
        Returns:
            Dataset instance
        """
        data_dir = Path(data_dir)
        
        # VQA-RAD has trainset.json and testset.json
        if split in ["train", "val"]:
            data_file = data_dir / "trainset.json"
        else:
            data_file = data_dir / "testset.json"
        
        image_dir = data_dir / "images"
        
        return cls(
            data_file=str(data_file),
            image_dir=str(image_dir),
            split=split,
            **kwargs
        )


class SLAKEDataset(MedicalVQADataset):
    """SLAKE specific dataset loader."""
    
    @classmethod
    def from_slake(
        cls,
        data_dir: str,
        split: str = "train",
        language: str = "en",
        **kwargs
    ) -> "SLAKEDataset":
        """
        Load SLAKE dataset.
        
        Args:
            data_dir: SLAKE data directory
            split: train/val/test
            language: en or zh
            **kwargs: Additional arguments
            
        Returns:
            Dataset instance
        """
        data_dir = Path(data_dir)
        
        # SLAKE structure
        data_file = data_dir / f"{split}.json"
        image_dir = data_dir / "imgs"
        
        return cls(
            data_file=str(data_file),
            image_dir=str(image_dir),
            split=split,
            **kwargs
        )


class PathVQADataset(MedicalVQADataset):
    """PathVQA specific dataset loader."""
    
    @classmethod
    def from_pathvqa(
        cls,
        data_dir: str,
        split: str = "train",
        **kwargs
    ) -> "PathVQADataset":
        """Load PathVQA dataset."""
        data_dir = Path(data_dir)
        
        data_file = data_dir / f"pvqa_{split}.json"
        image_dir = data_dir / "images" / split
        
        return cls(
            data_file=str(data_file),
            image_dir=str(image_dir),
            split=split,
            **kwargs
        )


class UnifiedMedicalVQADataset(Dataset):
    """
    Unified dataset that combines multiple medical VQA datasets.
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        sampling_weights: Optional[List[float]] = None
    ):
        """
        Initialize unified dataset.
        
        Args:
            datasets: List of individual datasets
            sampling_weights: Optional weights for sampling from each dataset
        """
        self.datasets = datasets
        self.sampling_weights = sampling_weights
        
        # Calculate cumulative lengths
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)
        
        self.total_length = total
        logger.info(f"Created unified dataset with {self.total_length} total samples")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict:
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                # Calculate local index
                local_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][local_idx]
        
        raise IndexError(f"Index {idx} out of range")


def create_data_loaders(
    train_file: str,
    val_file: str,
    test_file: str,
    image_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    processor: Optional[AutoProcessor] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_file: Path to training data JSON
        val_file: Path to validation data JSON
        test_file: Path to test data JSON
        image_dir: Image directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        processor: HuggingFace processor
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MedicalVQADataset(
        data_file=train_file,
        image_dir=image_dir,
        split="train",
        processor=processor,
        **kwargs
    )
    
    val_dataset = MedicalVQADataset(
        data_file=val_file,
        image_dir=image_dir,
        split="val",
        processor=processor,
        **kwargs
    )
    
    test_dataset = MedicalVQADataset(
        data_file=test_file,
        image_dir=image_dir,
        split="test",
        processor=processor,
        **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def prepare_unified_dataset(
    data_sources: Dict[str, str],
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Prepare unified dataset from multiple sources.
    
    Args:
        data_sources: Dict mapping dataset name to directory
        output_dir: Output directory for processed data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_file, val_file, test_file) paths
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # Process each data source
    # (Implementation would depend on specific dataset formats)
    
    # Shuffle and split
    random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    # Save splits
    train_file = output_dir / "train.json"
    val_file = output_dir / "val.json"
    test_file = output_dir / "test.json"
    
    for samples, file_path in [
        (train_samples, train_file),
        (val_samples, val_file),
        (test_samples, test_file)
    ]:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
    
    logger.info(f"Saved {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")
    
    return str(train_file), str(val_file), str(test_file)


if __name__ == "__main__":
    # Example usage
    print("Medical VQA Dataset Loader Module")
    print("=" * 50)
    
    # Example: Create a simple dataset for testing
    sample_data = [
        {
            "id": "1",
            "image": "sample.png",
            "question": "Is there a tumor visible?",
            "answer": "Yes",
            "modality": "CT",
            "question_type": "yes_no"
        },
        {
            "id": "2",
            "image": "sample2.png",
            "question": "What organ is shown?",
            "answer": "Lung",
            "modality": "X-ray",
            "question_type": "what"
        }
    ]
    
    print("Sample data structure:")
    print(json.dumps(sample_data[0], indent=2))
