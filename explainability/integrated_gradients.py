"""
Integrated Gradients Module
===========================
Path-integrated gradients for feature attribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from tqdm import tqdm
from loguru import logger


class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    
    Computes feature importance by integrating gradients along
    the path from a baseline to the input.
    """
    
    def __init__(
        self,
        model: nn.Module,
        multiply_by_inputs: bool = True,
        batch_size: int = 4,
    ):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: Model to explain
            multiply_by_inputs: Multiply gradients by input difference
            batch_size: Batch size for integration steps
        """
        self.model = model
        self.multiply_by_inputs = multiply_by_inputs
        self.batch_size = batch_size
    
    def __call__(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        baselines: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        internal_batch_size: int = 4,
        return_convergence_delta: bool = False,
        **model_kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Compute integrated gradients attribution.
        
        Args:
            inputs: Input tensor [B, C, H, W]
            target: Target class (None = use predicted class)
            baselines: Baseline tensor (None = zero baseline)
            n_steps: Number of integration steps
            internal_batch_size: Batch size for processing steps
            return_convergence_delta: Return convergence delta
            **model_kwargs: Additional model kwargs
            
        Returns:
            Attribution tensor and optionally convergence delta
        """
        # Set baseline
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        
        # Get device
        device = inputs.device
        
        # Get target class if not specified
        if target is None:
            with torch.no_grad():
                if 'pixel_values' not in model_kwargs:
                    model_kwargs['pixel_values'] = inputs
                outputs = self.model(**model_kwargs)
                logits = outputs.get('logits', outputs)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                target = logits.argmax(dim=-1)
        
        # Compute path integral
        attributions = self._compute_integral(
            inputs=inputs,
            baselines=baselines,
            target=target,
            n_steps=n_steps,
            model_kwargs=model_kwargs,
        )
        
        # Multiply by input difference
        if self.multiply_by_inputs:
            attributions = attributions * (inputs - baselines)
        
        if return_convergence_delta:
            delta = self._compute_convergence_delta(
                inputs=inputs,
                baselines=baselines,
                attributions=attributions,
                target=target,
                model_kwargs=model_kwargs,
            )
            return attributions, delta
        
        return attributions
    
    def _compute_integral(
        self,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        target: Union[int, torch.Tensor],
        n_steps: int,
        model_kwargs: Dict,
    ) -> torch.Tensor:
        """Compute integral using Riemann sum."""
        batch_size = inputs.shape[0]
        
        # Create scaled inputs along path
        scaled_inputs = []
        for step in range(n_steps):
            alpha = step / n_steps
            scaled = baselines + alpha * (inputs - baselines)
            scaled_inputs.append(scaled)
        
        # Stack all scaled inputs
        all_scaled = torch.stack(scaled_inputs, dim=0)  # [n_steps, B, C, H, W]
        all_scaled = all_scaled.reshape(-1, *inputs.shape[1:])  # [n_steps*B, C, H, W]
        
        # Compute gradients for all steps
        all_grads = []
        
        for i in range(0, len(all_scaled), self.batch_size):
            batch = all_scaled[i:i + self.batch_size].requires_grad_(True)
            
            # Forward pass
            kwargs = model_kwargs.copy()
            kwargs['pixel_values'] = batch
            outputs = self.model(**kwargs)
            
            logits = outputs.get('logits', outputs)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            
            # Create target tensor
            batch_targets = target.repeat(len(batch) // batch_size) if isinstance(target, torch.Tensor) else target
            
            # Compute gradients
            if isinstance(batch_targets, int):
                score = logits[:, batch_targets].sum()
            else:
                score = torch.gather(logits, 1, batch_targets.unsqueeze(-1)).sum()
            
            grads = torch.autograd.grad(score, batch, create_graph=False)[0]
            all_grads.append(grads.detach())
        
        # Concatenate and reshape
        all_grads = torch.cat(all_grads, dim=0)
        all_grads = all_grads.reshape(n_steps, batch_size, *inputs.shape[1:])
        
        # Average gradients (Riemann sum)
        avg_grads = all_grads.mean(dim=0)
        
        return avg_grads
    
    def _compute_convergence_delta(
        self,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        attributions: torch.Tensor,
        target: Union[int, torch.Tensor],
        model_kwargs: Dict,
    ) -> float:
        """Compute convergence delta (completeness axiom check)."""
        with torch.no_grad():
            # Get predictions for inputs
            kwargs = model_kwargs.copy()
            kwargs['pixel_values'] = inputs
            outputs_input = self.model(**kwargs)
            logits_input = outputs_input.get('logits', outputs_input)
            if logits_input.dim() == 3:
                logits_input = logits_input[:, -1, :]
            
            # Get predictions for baselines
            kwargs['pixel_values'] = baselines
            outputs_baseline = self.model(**kwargs)
            logits_baseline = outputs_baseline.get('logits', outputs_baseline)
            if logits_baseline.dim() == 3:
                logits_baseline = logits_baseline[:, -1, :]
            
            # Compute output difference
            if isinstance(target, int):
                output_diff = logits_input[:, target] - logits_baseline[:, target]
            else:
                output_diff = torch.gather(logits_input, 1, target.unsqueeze(-1)) - \
                             torch.gather(logits_baseline, 1, target.unsqueeze(-1))
            
            # Sum of attributions
            attr_sum = attributions.sum(dim=tuple(range(1, attributions.dim())))
            
            # Delta
            delta = (output_diff.squeeze() - attr_sum).abs().mean().item()
        
        return delta


class LayerIntegratedGradients(IntegratedGradients):
    """
    Layer-wise Integrated Gradients.
    
    Computes attributions for intermediate layer outputs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer: nn.Module,
        multiply_by_inputs: bool = True,
    ):
        """
        Initialize Layer Integrated Gradients.
        
        Args:
            model: Model to explain
            layer: Target layer
            multiply_by_inputs: Multiply by input difference
        """
        super().__init__(model, multiply_by_inputs)
        self.layer = layer
        self.layer_output = None
        self.layer_input = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks."""
        def forward_hook(module, input, output):
            self.layer_input = input[0] if isinstance(input, tuple) else input
            self.layer_output = output
        
        self.layer.register_forward_hook(forward_hook)


class ExpectedGradients(IntegratedGradients):
    """
    Expected Gradients (Shapley-based attribution).
    
    Approximates Shapley values by averaging integrated gradients
    over multiple baselines sampled from a reference distribution.
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_dataset: Optional[torch.utils.data.Dataset] = None,
        n_references: int = 10,
    ):
        """
        Initialize Expected Gradients.
        
        Args:
            model: Model to explain
            reference_dataset: Dataset to sample references from
            n_references: Number of reference samples
        """
        super().__init__(model)
        self.reference_dataset = reference_dataset
        self.n_references = n_references
    
    def __call__(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        n_steps: int = 50,
        **model_kwargs
    ) -> torch.Tensor:
        """Compute expected gradients."""
        all_attributions = []
        
        for _ in range(self.n_references):
            # Sample baseline
            if self.reference_dataset is not None:
                idx = np.random.randint(len(self.reference_dataset))
                baseline = self.reference_dataset[idx]
                if isinstance(baseline, dict):
                    baseline = baseline.get('pixel_values', baseline.get('image'))
                baseline = baseline.unsqueeze(0).to(inputs.device)
                if baseline.shape != inputs.shape:
                    baseline = baseline.expand_as(inputs)
            else:
                baseline = torch.randn_like(inputs)
            
            # Compute integrated gradients
            attrs = super().__call__(
                inputs=inputs,
                target=target,
                baselines=baseline,
                n_steps=n_steps,
                **model_kwargs
            )
            all_attributions.append(attrs)
        
        # Average over baselines
        return torch.stack(all_attributions).mean(dim=0)


class GradientSHAP(IntegratedGradients):
    """
    GradientSHAP attribution method.
    
    Approximates SHAP values using gradients with stochastic baselines.
    """
    
    def __init__(
        self,
        model: nn.Module,
        stdev: float = 0.1,
        n_samples: int = 25,
    ):
        """
        Initialize GradientSHAP.
        
        Args:
            model: Model to explain
            stdev: Standard deviation for noise
            n_samples: Number of samples
        """
        super().__init__(model)
        self.stdev = stdev
        self.n_samples = n_samples
    
    def __call__(
        self,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        target: Optional[int] = None,
        **model_kwargs
    ) -> torch.Tensor:
        """Compute GradientSHAP attributions."""
        all_attributions = []
        
        for _ in range(self.n_samples):
            # Random point along baseline-input path
            rand_point = baselines + torch.rand(1).to(inputs.device) * (inputs - baselines)
            
            # Add noise
            rand_point = rand_point + torch.randn_like(rand_point) * self.stdev
            rand_point.requires_grad_(True)
            
            # Forward pass
            kwargs = model_kwargs.copy()
            kwargs['pixel_values'] = rand_point
            outputs = self.model(**kwargs)
            
            logits = outputs.get('logits', outputs)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            
            # Get target
            if target is None:
                target = logits.argmax(dim=-1)
            
            # Backward
            if isinstance(target, int):
                score = logits[:, target].sum()
            else:
                score = torch.gather(logits, 1, target.unsqueeze(-1)).sum()
            
            grads = torch.autograd.grad(score, rand_point)[0]
            
            attribution = grads * (inputs - baselines)
            all_attributions.append(attribution)
        
        return torch.stack(all_attributions).mean(dim=0)


class MedicalIntegratedGradients:
    """
    Medical-specific Integrated Gradients.
    
    Provides domain-specific baselines and interpretations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        modality_baselines: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize Medical Integrated Gradients.
        
        Args:
            model: Medical VQA model
            modality_baselines: Baseline images per modality
        """
        self.ig = IntegratedGradients(model)
        self.modality_baselines = modality_baselines or {}
    
    def attribute(
        self,
        image: torch.Tensor,
        modality: str = "xray",
        target: Optional[int] = None,
        n_steps: int = 50,
        **model_kwargs
    ) -> Dict:
        """
        Compute attribution with modality-specific handling.
        
        Args:
            image: Input image
            modality: Imaging modality
            target: Target class
            n_steps: Integration steps
            **model_kwargs: Model kwargs
            
        Returns:
            Attribution results
        """
        # Get modality-specific baseline
        if modality.lower() in self.modality_baselines:
            baseline = self.modality_baselines[modality.lower()]
        else:
            # Use modality-appropriate baseline
            if modality.lower() in ['xray', 'ct']:
                # Black baseline (air)
                baseline = torch.zeros_like(image)
            elif modality.lower() == 'mri':
                # Gray baseline
                baseline = torch.full_like(image, 0.5)
            else:
                baseline = torch.zeros_like(image)
        
        # Compute attributions
        attributions, delta = self.ig(
            inputs=image,
            target=target,
            baselines=baseline,
            n_steps=n_steps,
            return_convergence_delta=True,
            **model_kwargs
        )
        
        # Analyze attribution pattern
        analysis = self._analyze_attributions(attributions, modality)
        
        return {
            'attributions': attributions,
            'convergence_delta': delta,
            'analysis': analysis,
            'modality': modality,
        }
    
    def _analyze_attributions(
        self,
        attributions: torch.Tensor,
        modality: str,
    ) -> Dict:
        """Analyze attribution patterns."""
        attr_np = attributions.detach().cpu().numpy()
        
        # Summarize channels
        attr_sum = np.abs(attr_np).sum(axis=1)  # [B, H, W]
        
        # Statistics
        stats = {
            'mean': float(attr_sum.mean()),
            'std': float(attr_sum.std()),
            'max': float(attr_sum.max()),
            'positive_ratio': float((attr_sum > 0).mean()),
        }
        
        # Find important regions
        threshold = stats['mean'] + stats['std']
        important_pixels = attr_sum > threshold
        stats['important_region_ratio'] = float(important_pixels.mean())
        
        return stats


def visualize_integrated_gradients(
    image: np.ndarray,
    attributions: np.ndarray,
    method: str = 'overlay',
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Visualize integrated gradients.
    
    Args:
        image: Original image [H, W, 3]
        attributions: Attribution values [C, H, W] or [H, W]
        method: Visualization method ('overlay', 'heatmap', 'mask')
        alpha: Transparency
        
    Returns:
        Visualization image
    """
    import cv2
    
    # Process attributions
    if attributions.ndim == 3:
        attributions = np.abs(attributions).sum(axis=0)
    
    # Normalize
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
    
    if method == 'overlay':
        # Color heatmap overlay
        heatmap = cv2.applyColorMap(
            (attributions * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    elif method == 'heatmap':
        # Pure heatmap
        result = cv2.applyColorMap(
            (attributions * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
    
    elif method == 'mask':
        # Masked image
        mask = attributions[:, :, np.newaxis]
        result = (image * mask).astype(np.uint8)
    
    else:
        result = image
    
    return result


if __name__ == "__main__":
    # Test Integrated Gradients
    print("Testing Integrated Gradients...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 100)
        
        def forward(self, pixel_values):
            x = F.relu(self.conv(pixel_values))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return {'logits': self.fc(x)}
    
    model = DummyModel()
    ig = IntegratedGradients(model)
    
    # Test input
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    attributions = ig(dummy_input, target=5, n_steps=10)
    
    print(f"Attribution shape: {attributions.shape}")
    print("Integrated Gradients test passed!")
