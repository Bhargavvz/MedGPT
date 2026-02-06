"""
Evaluation Pipeline Module
==========================
End-to-end evaluation for Medical VQA.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from loguru import logger

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.metrics import VQAEvaluator, compute_vqa_metrics


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'num_samples': len(self.predictions),
                'num_errors': len(self.errors),
                'examples': self.examples[:20],  # Save first 20 examples
            }, f, indent=2)
    
    def summary(self) -> str:
        """Get summary string."""
        lines = ["Evaluation Summary", "=" * 40]
        lines.append(f"Total samples: {len(self.predictions)}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append("")
        lines.append("Metrics:")
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class EvaluationPipeline:
    """
    End-to-end evaluation pipeline for Medical VQA.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 64,
        batch_size: int = 16,
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            model: VQA model
            tokenizer: Tokenizer
            device: Device to use
            max_new_tokens: Max new tokens for generation
            batch_size: Evaluation batch size
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        self.model.to(self.device)
        self.model.eval()
        
        self.evaluator = VQAEvaluator(tokenizer=tokenizer)
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_explanations: bool = False,
        save_examples: int = 100,
    ) -> EvaluationResults:
        """
        Run evaluation on dataset.
        
        Args:
            dataloader: Data loader
            compute_explanations: Generate explanations
            save_examples: Number of examples to save
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {len(dataloader)} batches...")
        
        results = EvaluationResults()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Move to device
                batch = self._prepare_batch(batch)
                
                # Generate predictions
                outputs = self.model.generate(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    max_new_tokens=self.max_new_tokens,
                    generate_explanation=compute_explanations,
                )
                
                # Decode predictions
                pred_ids = outputs.get('generated_ids', outputs)
                predictions = self.tokenizer.batch_decode(
                    pred_ids,
                    skip_special_tokens=True
                )
                
                # Get references
                references = batch.get('answer', batch.get('labels', []))
                if isinstance(references, torch.Tensor):
                    references = self.tokenizer.batch_decode(
                        references,
                        skip_special_tokens=True
                    )
                
                # Store results
                for i, (pred, ref) in enumerate(zip(predictions, references)):
                    results.predictions.append(pred)
                    results.references.append(ref)
                    
                    # Save examples
                    if len(results.examples) < save_examples:
                        example = {
                            'prediction': pred,
                            'reference': ref,
                            'question': batch.get('question', [''])[i] if 'question' in batch else '',
                        }
                        if compute_explanations and 'explanation_ids' in outputs:
                            exp_text = self.tokenizer.decode(
                                outputs['explanation_ids'][i],
                                skip_special_tokens=True
                            )
                            example['explanation'] = exp_text
                        
                        results.examples.append(example)
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                results.errors.append({
                    'batch_idx': batch_idx,
                    'error': str(e)
                })
        
        # Compute metrics
        results.metrics = compute_vqa_metrics(
            results.predictions,
            results.references
        )
        
        logger.info(results.summary())
        
        return results
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Move batch to device."""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        return prepared
    
    def evaluate_single(
        self,
        image: torch.Tensor,
        question: str,
        reference: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate single example.
        
        Args:
            image: Image tensor
            question: Question text
            reference: Optional ground truth
            
        Returns:
            Evaluation result
        """
        # Tokenize question
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Generate
        outputs = self.model.generate(
            pixel_values=image.unsqueeze(0).to(self.device),
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_new_tokens=self.max_new_tokens,
            generate_explanation=True,
        )
        
        # Decode
        prediction = self.tokenizer.decode(
            outputs['generated_ids'][0],
            skip_special_tokens=True
        )
        
        result = {
            'question': question,
            'prediction': prediction,
        }
        
        if 'explanation_ids' in outputs:
            result['explanation'] = self.tokenizer.decode(
                outputs['explanation_ids'][0],
                skip_special_tokens=True
            )
        
        if reference:
            result['reference'] = reference
            result['metrics'] = compute_vqa_metrics([prediction], [reference])
        
        return result


class AblationStudy:
    """
    Ablation study framework for Medical VQA.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        dataloader: DataLoader,
        device: str = "cuda",
    ):
        """
        Initialize ablation study.
        
        Args:
            model: Base VQA model
            tokenizer: Tokenizer
            dataloader: Evaluation data
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.device = device
        
        self.results = {}
    
    def run_ablation(
        self,
        ablation_name: str,
        model_modifier: callable,
    ) -> Dict[str, float]:
        """
        Run single ablation.
        
        Args:
            ablation_name: Name of ablation
            model_modifier: Function to modify model
            
        Returns:
            Ablation metrics
        """
        logger.info(f"Running ablation: {ablation_name}")
        
        # Modify model
        modified_model = model_modifier(self.model)
        
        # Evaluate
        pipeline = EvaluationPipeline(
            model=modified_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        results = pipeline.evaluate(self.dataloader)
        
        self.results[ablation_name] = results.metrics
        
        return results.metrics
    
    def run_all_ablations(self) -> Dict[str, Dict[str, float]]:
        """Run all standard ablations."""
        ablations = {
            'full_model': lambda m: m,
            'no_knowledge': self._ablate_knowledge,
            'frozen_vision': self._ablate_vision,
            'no_fusion': self._ablate_fusion,
            'no_explanation': self._ablate_explanation,
        }
        
        for name, modifier in ablations.items():
            self.run_ablation(name, modifier)
        
        return self.results
    
    def _ablate_knowledge(self, model):
        """Disable knowledge encoder."""
        if hasattr(model, 'knowledge_encoder'):
            model._knowledge_enabled = False
        return model
    
    def _ablate_vision(self, model):
        """Freeze vision encoder."""
        if hasattr(model, 'vision_encoder'):
            for param in model.vision_encoder.parameters():
                param.requires_grad = False
        return model
    
    def _ablate_fusion(self, model):
        """Disable fusion module."""
        if hasattr(model, 'fusion'):
            model._fusion_enabled = False
        return model
    
    def _ablate_explanation(self, model):
        """Disable explanation generation."""
        model.generate_rationale = False
        return model
    
    def comparison_table(self) -> str:
        """Generate comparison table."""
        if not self.results:
            return "No ablation results available."
        
        # Get all metric names
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        
        # Build table
        lines = []
        header = "| Ablation | " + " | ".join(sorted(all_metrics)) + " |"
        separator = "|" + "-" * (len(header) - 2) + "|"
        
        lines.append(header)
        lines.append(separator)
        
        for ablation, metrics in self.results.items():
            row = f"| {ablation} |"
            for metric in sorted(all_metrics):
                value = metrics.get(metric, 0.0)
                row += f" {value:.4f} |"
            lines.append(row)
        
        return "\n".join(lines)
    
    def save_results(self, path: str):
        """Save ablation results."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)


if __name__ == "__main__":
    print("Evaluation pipeline module loaded successfully")
    
    # Example usage
    results = EvaluationResults()
    results.metrics = {'accuracy': 0.75, 'f1': 0.72, 'bleu_1': 0.68}
    results.predictions = ['yes', 'pneumonia', 'left lung']
    results.references = ['yes', 'pneumonia', 'the left lung']
    
    print(results.summary())
