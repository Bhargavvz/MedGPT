"""
Trainer Module
==============
Custom trainer for Medical VQA with LoRA fine-tuning.
"""

import os
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from loguru import logger

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from training.loss_functions import VQALoss


@dataclass
class TrainingArguments:
    """Training configuration arguments."""
    output_dir: str = "./checkpoints"
    num_epochs: int = 15
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    freeze_vision_epochs: int = 3
    
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    eval_strategy: str = "steps"
    eval_steps: int = 500
    
    logging_steps: int = 10
    logging_dir: str = "./logs"
    
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # Loss weights
    answer_loss_weight: float = 1.0
    attention_alignment_weight: float = 0.1
    knowledge_grounding_weight: float = 0.2
    rationale_generation_weight: float = 0.3


class MedicalVQATrainer:
    """
    Custom trainer for Medical VQA.
    
    Features:
    - LoRA fine-tuning
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Progressive vision encoder unfreezing
    - Multi-objective loss
    - TensorBoard logging
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Tuple[torch.optim.Optimizer, Optional[object]] = (None, None),
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            compute_metrics: Metrics computation function
            optimizers: Optional (optimizer, scheduler) tuple
        """
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = optimizers
        if self.optimizer is None:
            self.optimizer, self.scheduler = self._create_optimizer_and_scheduler()
        
        # Setup loss function
        self.loss_fn = VQALoss(
            answer_loss_weight=args.answer_loss_weight,
            attention_alignment_weight=args.attention_alignment_weight,
            knowledge_grounding_weight=args.knowledge_grounding_weight,
            rationale_generation_weight=args.rationale_generation_weight,
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if args.fp16 else None
        self.autocast_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
        
        # Setup logging
        self.logging_dir = Path(args.logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.logging_dir))
        else:
            self.writer = None
            logger.warning("TensorBoard not available")
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('-inf')
        
        # Set seed
        self._set_seed(args.seed)
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        if eval_dataloader:
            logger.info(f"Eval samples: {len(eval_dataloader.dataset)}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, object]:
        """Create optimizer and learning rate scheduler."""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        total_steps = num_update_steps_per_epoch * self.args.num_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine decay scheduler
        min_lr = self.args.learning_rate * self.args.min_lr_ratio
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr
        )
        
        # Combined scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        logger.info(f"Created optimizer with LR={self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        return optimizer, scheduler
    
    def train(self) -> Dict:
        """
        Run full training loop.
        
        Returns:
            Training metrics
        """
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if self.args.resume_from_checkpoint:
            self._load_checkpoint(self.args.resume_from_checkpoint)
        
        train_losses = []
        
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.current_epoch = epoch
            
            # Progressive unfreezing
            if epoch == self.args.freeze_vision_epochs:
                logger.info("Unfreezing vision encoder")
                if hasattr(self.model, 'unfreeze_vision_encoder'):
                    self.model.unfreeze_vision_encoder()
            
            # Train one epoch
            epoch_loss = self._train_epoch(epoch)
            train_losses.append(epoch_loss)
            
            # Evaluate
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                
                # Log eval metrics
                if self.writer:
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f"eval/{key}", value, self.global_step)
                
                # Save best model
                if eval_metrics.get('accuracy', 0) > self.best_metric:
                    self.best_metric = eval_metrics.get('accuracy', 0)
                    self._save_checkpoint("best_model")
        
        # Final save
        self._save_checkpoint("final_model")
        
        if self.writer:
            self.writer.close()
        
        return {
            'train_losses': train_losses,
            'best_metric': self.best_metric,
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.args.num_epochs}",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Handle both tokenized and basic batch formats
            if 'input_ids' not in batch:
                # Dataset should provide tokenized data, but handle legacy format
                pixel_values = batch.get('image', batch.get('pixel_values'))
                if pixel_values is None:
                    continue
                    
                # Use model's tokenizer to create proper inputs
                batch_size = pixel_values.shape[0]
                seq_len = 64
                
                # If we have the model's tokenizer, encode a placeholder
                if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                    tokenizer = self.model.tokenizer
                    placeholder = "What is shown in this medical image?"
                    encoded = tokenizer(
                        [placeholder] * batch_size,
                        max_length=seq_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    batch['input_ids'] = encoded['input_ids'].to(self.device)
                    batch['attention_mask'] = encoded['attention_mask'].to(self.device)
                else:
                    batch['input_ids'] = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
                    batch['attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
                
                batch['pixel_values'] = pixel_values
                
                # Create labels if not present
                if 'labels' not in batch:
                    batch['labels'] = batch['input_ids'].clone()
            
            # Forward pass with mixed precision
            with autocast(dtype=self.autocast_dtype, enabled=self.args.fp16 or self.args.bf16):
                # Handle knowledge_snippet - could be a list of strings from batch collation
                knowledge = batch.get('knowledge_texts', batch.get('knowledge_snippet'))
                if isinstance(knowledge, (list, tuple)) and len(knowledge) > 0:
                    # Filter out empty strings
                    knowledge = [k for k in knowledge if k] or None
                elif isinstance(knowledge, str):
                    knowledge = [knowledge] if knowledge else None
                else:
                    knowledge = None
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch.get('pixel_values', batch.get('image')),
                    labels=batch.get('labels'),
                    knowledge_texts=knowledge,
                    return_attention=True,
                )
                
                # Compute loss
                losses = self.loss_fn(
                    logits=outputs['logits'],
                    labels=batch['labels'],
                    attention_weights=outputs.get('attention_weights'),
                    knowledge_gate_values=outputs.get('attention_weights', {}).get('knowledge_gate'),
                    rationale_logits=outputs.get('explanation', {}).get('logits') if 'explanation' in outputs else None,
                    rationale_labels=batch.get('rationale_labels'),
                )
                
                loss = losses['total_loss'] / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.args.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                    
                    if self.writer:
                        self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                        self.writer.add_scalar("train/learning_rate", lr, self.global_step)
                        
                        for key, value in losses.items():
                            if key != 'total_loss':
                                self.writer.add_scalar(f"train/{key}", value.item(), self.global_step)
                
                # Checkpoint saving
                if self.args.save_strategy == "steps" and self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")
                
                # Evaluation
                if self.args.eval_strategy == "steps" and self.global_step % self.args.eval_steps == 0:
                    if self.eval_dataloader is not None:
                        eval_metrics = self.evaluate()
                        self.model.train()  # Switch back to train mode
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self) -> Dict:
        """
        Run evaluation.
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = self._prepare_batch(batch)
                
                # Handle basic batch format (legacy)
                if 'input_ids' not in batch:
                    pixel_values = batch.get('image', batch.get('pixel_values'))
                    if pixel_values is None:
                        continue
                    batch_size = pixel_values.shape[0]
                    seq_len = 64
                    
                    if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                        tokenizer = self.model.tokenizer
                        placeholder = "What is shown in this medical image?"
                        encoded = tokenizer(
                            [placeholder] * batch_size,
                            max_length=seq_len,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        batch['input_ids'] = encoded['input_ids'].to(self.device)
                        batch['attention_mask'] = encoded['attention_mask'].to(self.device)
                    else:
                        batch['input_ids'] = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
                        batch['attention_mask'] = torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device)
                    
                    batch['pixel_values'] = pixel_values
                    if 'labels' not in batch:
                        batch['labels'] = batch['input_ids'].clone()
                
                with autocast(dtype=self.autocast_dtype, enabled=self.args.fp16 or self.args.bf16):
                    # Handle knowledge
                    knowledge = batch.get('knowledge_texts', batch.get('knowledge_snippet'))
                    if isinstance(knowledge, (list, tuple)) and len(knowledge) > 0:
                        knowledge = [k for k in knowledge if k] or None
                    elif isinstance(knowledge, str):
                        knowledge = [knowledge] if knowledge else None
                    else:
                        knowledge = None
                    
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch.get('pixel_values', batch.get('image')),
                        labels=batch.get('labels'),
                        knowledge_texts=knowledge,
                    )
                    
                    losses = self.loss_fn(
                        logits=outputs['logits'],
                        labels=batch['labels'],
                    )
                    
                    total_loss += losses['total_loss'].item()
                
                # Get predictions
                predictions = outputs['logits'].argmax(dim=-1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())
        
        # Compute metrics
        metrics = {
            'eval_loss': total_loss / len(self.eval_dataloader),
        }
        
        if self.compute_metrics:
            custom_metrics = self.compute_metrics(all_predictions, all_labels)
            metrics.update(custom_metrics)
        else:
            # Default accuracy
            correct = sum(1 for p, l in zip(all_predictions, all_labels) 
                         if p == l and l != -100)
            total = sum(1 for l in all_labels if l != -100)
            metrics['accuracy'] = correct / max(total, 1)
        
        logger.info(f"Eval metrics: {metrics}")
        
        return metrics
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Move batch to device."""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Save args
        with open(checkpoint_dir / "training_args.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model state
        if (checkpoint_dir / "model.pt").exists():
            self.model.load_state_dict(torch.load(checkpoint_dir / "model.pt"))
        
        # Load training state
        if (checkpoint_dir / "training_state.pt").exists():
            training_state = torch.load(checkpoint_dir / "training_state.pt")
            self.global_step = training_state['global_step']
            self.current_epoch = training_state['current_epoch']
            self.best_metric = training_state['best_metric']
            self.optimizer.load_state_dict(training_state['optimizer_state'])
            if self.scheduler and training_state['scheduler_state']:
                self.scheduler.load_state_dict(training_state['scheduler_state'])
        
        logger.info(f"Resumed from checkpoint: {checkpoint_dir}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only save_total_limit most recent."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1])
        )
        
        if len(checkpoints) > self.args.save_total_limit:
            for checkpoint in checkpoints[:-self.args.save_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")


if __name__ == "__main__":
    # Example usage
    print("Trainer module loaded successfully")
    
    # Create dummy args
    args = TrainingArguments(
        output_dir="./test_checkpoints",
        num_epochs=1,
        batch_size=2,
        learning_rate=2e-5,
    )
    
    print(f"Training args: {vars(args)}")
