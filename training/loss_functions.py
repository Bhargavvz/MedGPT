"""
Loss Functions Module
=====================
Custom loss functions for Medical VQA training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class VQALoss(nn.Module):
    """
    Main VQA loss combining multiple objectives.
    
    L_total = L_answer + λ₁*L_attention + λ₂*L_knowledge + λ₃*L_rationale
    """
    
    def __init__(
        self,
        answer_loss_weight: float = 1.0,
        attention_alignment_weight: float = 0.1,
        knowledge_grounding_weight: float = 0.2,
        rationale_generation_weight: float = 0.3,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Initialize VQA loss.
        
        Args:
            answer_loss_weight: Weight for answer prediction loss
            attention_alignment_weight: Weight for attention alignment loss
            knowledge_grounding_weight: Weight for knowledge grounding loss
            rationale_generation_weight: Weight for rationale generation loss
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        
        self.answer_loss_weight = answer_loss_weight
        self.attention_alignment_weight = attention_alignment_weight
        self.knowledge_grounding_weight = knowledge_grounding_weight
        self.rationale_generation_weight = rationale_generation_weight
        self.ignore_index = ignore_index
        
        # Answer prediction loss
        self.answer_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # Rationale generation loss
        self.rationale_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: Optional[Dict[str, torch.Tensor]] = None,
        knowledge_gate_values: Optional[torch.Tensor] = None,
        rationale_logits: Optional[torch.Tensor] = None,
        rationale_labels: Optional[torch.Tensor] = None,
        attention_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            logits: Answer prediction logits [B, L, V]
            labels: Ground truth labels [B, L]
            attention_weights: Vision-text attention weights
            knowledge_gate_values: Knowledge gating values
            rationale_logits: Rationale generation logits
            rationale_labels: Rationale ground truth
            attention_targets: Target attention distribution
            
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        
        # =====================================================================
        # Answer Prediction Loss
        # =====================================================================
        answer_loss = self.answer_loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        losses['answer_loss'] = answer_loss
        
        # =====================================================================
        # Attention Alignment Loss
        # =====================================================================
        if attention_weights is not None and attention_targets is not None:
            attention_loss = self._compute_attention_loss(
                attention_weights,
                attention_targets
            )
            losses['attention_loss'] = attention_loss
        else:
            attention_loss = torch.tensor(0.0, device=logits.device)
            losses['attention_loss'] = attention_loss
        
        # =====================================================================
        # Knowledge Grounding Loss
        # =====================================================================
        if knowledge_gate_values is not None:
            knowledge_loss = self._compute_knowledge_loss(knowledge_gate_values)
            losses['knowledge_loss'] = knowledge_loss
        else:
            knowledge_loss = torch.tensor(0.0, device=logits.device)
            losses['knowledge_loss'] = knowledge_loss
        
        # =====================================================================
        # Rationale Generation Loss
        # =====================================================================
        if rationale_logits is not None and rationale_labels is not None:
            rationale_loss = self.rationale_loss_fn(
                rationale_logits.view(-1, rationale_logits.size(-1)),
                rationale_labels.view(-1)
            )
            losses['rationale_loss'] = rationale_loss
        else:
            rationale_loss = torch.tensor(0.0, device=logits.device)
            losses['rationale_loss'] = rationale_loss
        
        # =====================================================================
        # Total Loss
        # =====================================================================
        total_loss = (
            self.answer_loss_weight * answer_loss +
            self.attention_alignment_weight * attention_loss +
            self.knowledge_grounding_weight * knowledge_loss +
            self.rationale_generation_weight * rationale_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_attention_loss(
        self,
        attention_weights: Dict[str, torch.Tensor],
        attention_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention alignment loss.
        
        Encourages model to attend to relevant image regions.
        """
        # Get vision-text attention from last layer
        key = list(attention_weights.keys())[0]
        attn = attention_weights[key]
        
        if attn is None:
            return torch.tensor(0.0)
        
        # Average over heads
        if len(attn.shape) == 4:
            attn = attn.mean(dim=1)
        
        # KL divergence between predicted and target attention
        attn_log = F.log_softmax(attn.flatten(1), dim=-1)
        target_dist = F.softmax(attention_targets.flatten(1), dim=-1)
        
        kl_loss = F.kl_div(attn_log, target_dist, reduction='batchmean')
        
        return kl_loss
    
    def _compute_knowledge_loss(
        self,
        gate_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge grounding loss.
        
        Encourages appropriate use of knowledge (not too sparse, not too uniform).
        """
        # Entropy regularization - encourage informative gating
        gate_flat = gate_values.view(-1)
        
        # Avoid log(0)
        gate_flat = gate_flat.clamp(min=1e-7, max=1 - 1e-7)
        
        # Binary entropy loss - push gates towards decisive values
        entropy = -gate_flat * torch.log(gate_flat) - (1 - gate_flat) * torch.log(1 - gate_flat)
        
        # Minimize entropy to encourage decisive gating
        return entropy.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Balancing factor
            gamma: Focusing parameter
            ignore_index: Index to ignore
            reduction: Reduction method
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions [B, C] or [B, L, C]
            targets: Targets [B] or [B, L]
            
        Returns:
            Focal loss value
        """
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        # Create mask for valid targets
        mask = targets != self.ignore_index
        
        # Filter
        inputs = inputs[mask]
        targets = targets[mask]
        
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for vision-language alignment.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature scaling
            label_smoothing: Label smoothing
        """
        super().__init__()
        
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            vision_features: Vision features [B, D]
            text_features: Text features [B, D]
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(vision_features, text_features.T) / self.temperature
        
        # Labels are diagonal (matching pairs)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric contrastive loss
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        
        return (loss_v2t + loss_t2v) / 2


class AnswerTypeLoss(nn.Module):
    """
    Loss function that handles different answer types differently.
    
    - Yes/No questions: Binary cross-entropy
    - Open-ended: Cross-entropy with label smoothing
    - Counting: MSE or ordinal loss
    """
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Initialize answer type loss.
        
        Args:
            num_classes: Number of answer classes
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Binary loss for yes/no
        self.binary_loss = nn.BCEWithLogitsLoss()
        
        # Multi-class loss for open-ended
        self.multiclass_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        answer_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute answer-type aware loss.
        
        Args:
            logits: Prediction logits
            labels: Ground truth labels
            answer_types: List of answer types per sample
            
        Returns:
            Combined loss
        """
        if answer_types is None:
            # Fall back to standard cross-entropy
            return self.multiclass_loss(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        
        batch_size = logits.shape[0]
        total_loss = 0.0
        count = 0
        
        for i, answer_type in enumerate(answer_types):
            if answer_type == 'yes_no':
                # Binary classification
                if logits[i].dim() > 1:
                    pred = logits[i, :2].mean()  # Assume first 2 classes are yes/no
                else:
                    pred = logits[i]
                target = labels[i].float().clamp(0, 1)
                total_loss += self.binary_loss(pred.unsqueeze(0), target.unsqueeze(0))
            else:
                # Multi-class
                total_loss += self.multiclass_loss(
                    logits[i:i+1].view(-1, logits.size(-1)),
                    labels[i:i+1].view(-1)
                )
            count += 1
        
        return total_loss / max(count, 1)


class MultiTaskVQALoss(nn.Module):
    """
    Multi-task loss for VQA with learnable task weights.
    """
    
    def __init__(
        self,
        num_tasks: int = 4,
        initial_weights: Optional[List[float]] = None,
    ):
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks
            initial_weights: Initial task weights
        """
        super().__init__()
        
        if initial_weights is None:
            initial_weights = [1.0] * num_tasks
        
        # Learnable log-variance for uncertainty weighting
        self.log_vars = nn.Parameter(
            torch.tensor([float(w) for w in initial_weights])
        )
    
    def forward(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-task loss with learned weights.
        
        Args:
            losses: Dictionary of individual losses
            
        Returns:
            Weighted total loss
        """
        loss_values = list(losses.values())[:len(self.log_vars)]
        
        total_loss = 0.0
        for i, loss in enumerate(loss_values):
            # Uncertainty weighting: L_i / (2 * σ_i^2) + log(σ_i)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    
    batch_size = 4
    seq_len = 64
    vocab_size = 32000
    
    # Create dummy data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test VQA loss
    vqa_loss = VQALoss(
        answer_loss_weight=1.0,
        attention_alignment_weight=0.1,
        knowledge_grounding_weight=0.2,
        rationale_generation_weight=0.3,
    )
    
    losses = vqa_loss(logits, labels)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Answer loss: {losses['answer_loss'].item():.4f}")
    
    # Test focal loss
    focal = FocalLoss(gamma=2.0)
    focal_loss_val = focal(logits.view(-1, vocab_size), labels.view(-1))
    print(f"Focal loss: {focal_loss_val.item():.4f}")
    
    # Test contrastive loss
    contrast = ContrastiveLoss(temperature=0.07)
    vision_feat = torch.randn(batch_size, 512)
    text_feat = torch.randn(batch_size, 512)
    contrast_loss_val = contrast(vision_feat, text_feat)
    print(f"Contrastive loss: {contrast_loss_val.item():.4f}")
    
    print("Loss functions test passed!")
