"""
Training Script
===============
Main training script for Medical VQA.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config, Config
from models import MedicalVQAModel, create_medical_vqa_model
from data import MedicalVQADataset, create_data_loaders
from training.trainer import MedicalVQATrainer, TrainingArguments
from evaluation.metrics import compute_vqa_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Medical VQA Model")
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    # Data
    parser.add_argument("--train_file", type=str, help="Training data file")
    parser.add_argument("--val_file", type=str, help="Validation data file")
    parser.add_argument("--image_dir", type=str, help="Image directory")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Model
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--no_quantization", action="store_true")
    
    # Other
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Run with minimal data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    
    return parser.parse_args()


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="100 MB",
        retention="30 days",
        level="INFO"
    )
    
    logger.info(f"Logging to {log_file}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("Medical VQA Training")
    logger.info("=" * 60)
    
    # Load configuration
    if Path(args.config).exists():
        config = get_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = Config()
        logger.warning("Using default configuration")
    
    # Override config with command line args
    if args.train_file:
        config.data.train_file = args.train_file
    if args.val_file:
        config.data.val_file = args.val_file
    if args.image_dir:
        config.data.raw_data_dir = args.image_dir
    if args.base_model:
        config.model.base_model = args.base_model
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.no_quantization:
        config.quantization.enabled = False
    
    # Log configuration
    logger.info(f"Base model: {config.model.base_model}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"LoRA: {config.lora.enabled}")
    logger.info(f"Quantization: {config.quantization.enabled}")
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available. Training will be slow.")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    if args.dry_run:
        # Create minimal dummy data for testing
        logger.info("Dry run mode - using minimal dummy data")
        train_dataset = create_dummy_dataset(num_samples=32)
        val_dataset = create_dummy_dataset(num_samples=8)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size if args.batch_size else 2,
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size if args.batch_size else 2
        )
    else:
        # Use actual data
        if not Path(config.data.train_file).exists():
            logger.error(f"Training file not found: {config.data.train_file}")
            logger.info("Please prepare your dataset or use --dry_run for testing")
            return
        
        train_dataloader, val_dataloader, _ = create_data_loaders(
            train_file=config.data.train_file,
            val_file=config.data.val_file,
            test_file=config.data.test_file,
            image_dir=config.data.raw_data_dir,
            batch_size=config.training.batch_size,
            num_workers=config.training.dataloader_num_workers,
        )
    
    logger.info(f"Train batches: {len(train_dataloader)}")
    logger.info(f"Val batches: {len(val_dataloader)}")
    
    # Create model
    logger.info("Creating model...")
    
    try:
        model = create_medical_vqa_model(config)
    except Exception as e:
        logger.warning(f"Could not create full model: {e}")
        logger.info("Creating simplified model for testing...")
        
        # Simplified model for testing
        model = MedicalVQAModel(
            base_model_name=config.model.base_model,
            use_lora=config.lora.enabled,
            use_quantization=False,  # Disable for testing
            generate_rationale=config.model.generate_rationale,
        )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        warmup_ratio=config.training.warmup_ratio,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        freeze_vision_epochs=config.training.unfreeze_vision_epoch,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        logging_dir=args.logging_dir,
        seed=args.seed,
        resume_from_checkpoint=args.resume,
        answer_loss_weight=config.loss.answer_loss_weight,
        attention_alignment_weight=config.loss.attention_alignment_weight,
        knowledge_grounding_weight=config.loss.knowledge_grounding_weight,
        rationale_generation_weight=config.loss.rationale_generation_weight,
    )
    
    # Create trainer
    trainer = MedicalVQATrainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        compute_metrics=compute_vqa_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    train_results = trainer.train()
    
    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best metric: {train_results['best_metric']:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("=" * 60)


def create_dummy_dataset(num_samples: int = 32):
    """Create dummy dataset for testing."""
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 32000, (64,)),
                'attention_mask': torch.ones(64),
                'pixel_values': torch.randn(3, 224, 224),
                'labels': torch.randint(0, 32000, (64,)),
                'knowledge_texts': "Pneumonia is a lung infection.",
            }
    
    return DummyDataset(num_samples)


if __name__ == "__main__":
    main()
