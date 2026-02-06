"""
Attention Visualization Module
==============================
Visualizes attention patterns in transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from loguru import logger


class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    
    Aggregates attention weights across layers to show
    how tokens attend to each other throughout the network.
    """
    
    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.1,
    ):
        """
        Initialize Attention Rollout.
        
        Args:
            model: Transformer model
            head_fusion: How to combine heads ('mean', 'max', 'min')
            discard_ratio: Ratio of lowest attention to discard
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        # Storage for attention maps
        self.attention_maps = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register attention hooks on transformer layers."""
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'get_attention_map'):
                    module.register_forward_hook(self._attention_hook)
                elif isinstance(module, nn.MultiheadAttention):
                    # Wrap MHA to capture attention
                    original_forward = module.forward
                    
                    def new_forward(query, key, value, *args, **kwargs):
                        # Force return of attention weights
                        kwargs['need_weights'] = True
                        kwargs['average_attn_weights'] = False
                        output, attn_weights = original_forward(query, key, value, *args, **kwargs)
                        self.attention_maps.append(attn_weights)
                        return output, attn_weights
                    
                    module.forward = new_forward
    
    def _attention_hook(self, module, input, output):
        """Hook to capture attention maps."""
        if hasattr(module, 'attention_weights'):
            self.attention_maps.append(module.attention_weights)
        elif isinstance(output, tuple) and len(output) > 1:
            self.attention_maps.append(output[1])
    
    def reset(self):
        """Reset attention maps."""
        self.attention_maps = []
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        **model_kwargs
    ) -> np.ndarray:
        """
        Compute attention rollout.
        
        Args:
            input_tensor: Input tensor
            **model_kwargs: Additional model kwargs
            
        Returns:
            Rolled out attention map
        """
        self.reset()
        
        # Forward pass
        with torch.no_grad():
            if 'pixel_values' not in model_kwargs:
                model_kwargs['pixel_values'] = input_tensor
            _ = self.model(**model_kwargs)
        
        if not self.attention_maps:
            logger.warning("No attention maps captured")
            return np.zeros((input_tensor.shape[0], 196))
        
        # Process attention maps
        result = self._rollout(self.attention_maps)
        
        return result
    
    def _rollout(
        self,
        attention_maps: List[torch.Tensor],
    ) -> np.ndarray:
        """Compute attention rollout across layers."""
        # Stack and process attention maps
        processed_maps = []
        
        for attn in attention_maps:
            if attn is None:
                continue
            
            # Ensure [B, H, N, N] format
            if attn.dim() == 3:
                attn = attn.unsqueeze(1)
            
            # Fuse heads
            if self.head_fusion == "mean":
                attn = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn = attn.max(dim=1)[0]
            elif self.head_fusion == "min":
                attn = attn.min(dim=1)[0]
            
            processed_maps.append(attn)
        
        if not processed_maps:
            return np.zeros((attention_maps[0].shape[0], 196))
        
        # Initialize with identity
        batch_size, num_tokens = processed_maps[0].shape[0], processed_maps[0].shape[1]
        result = torch.eye(num_tokens, device=processed_maps[0].device)
        result = result.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Rollout
        for attn in processed_maps:
            # Add residual connection
            attn = attn + torch.eye(num_tokens, device=attn.device).unsqueeze(0)
            
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Accumulate
            result = torch.bmm(attn, result)
        
        # Get CLS token attention to patches
        mask = result[:, 0, 1:]  # Exclude CLS token
        
        return mask.cpu().numpy()
    
    def visualize(
        self,
        image: np.ndarray,
        attention: np.ndarray,
        patch_size: int = 16,
    ) -> np.ndarray:
        """
        Overlay attention on image.
        
        Args:
            image: Original image (H, W, 3)
            attention: Attention values for patches
            patch_size: Size of image patches
            
        Returns:
            Visualization with attention overlay
        """
        h, w = image.shape[:2]
        
        # Reshape attention to 2D grid
        num_patches_side = int(np.sqrt(len(attention)))
        attention_2d = attention.reshape(num_patches_side, num_patches_side)
        
        # Resize to image size
        attention_resized = cv2.resize(
            attention_2d,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attention_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        
        return overlay


class CrossAttentionVisualization:
    """
    Visualize cross-attention between modalities.
    """
    
    def __init__(
        self,
        model: nn.Module,
    ):
        """
        Initialize cross-attention visualizer.
        
        Args:
            model: Model with cross-attention
        """
        self.model = model
        self.cross_attention_maps = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for cross-attention layers."""
        for name, module in self.model.named_modules():
            if 'cross' in name.lower() and 'attn' in name.lower():
                def hook_fn(n):
                    def hook(module, input, output):
                        if isinstance(output, tuple) and len(output) > 1:
                            self.cross_attention_maps[n] = output[1]
                    return hook
                
                module.register_forward_hook(hook_fn(name))
    
    def reset(self):
        """Reset attention maps."""
        self.cross_attention_maps = {}
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        **model_kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Get cross-attention maps.
        
        Args:
            input_tensor: Input tensor
            **model_kwargs: Model kwargs
            
        Returns:
            Dictionary of attention maps
        """
        self.reset()
        
        with torch.no_grad():
            if 'pixel_values' not in model_kwargs:
                model_kwargs['pixel_values'] = input_tensor
            _ = self.model(**model_kwargs)
        
        result = {}
        for name, attn in self.cross_attention_maps.items():
            if attn is not None:
                result[name] = attn.cpu().numpy()
        
        return result
    
    def visualize_vision_text(
        self,
        attention: np.ndarray,
        image: np.ndarray,
        tokens: List[str],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Visualize vision-text cross-attention.
        
        Args:
            attention: Attention matrix [num_text, num_vision]
            image: Original image
            tokens: Text tokens
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Attention matrix
        im = axes[0].imshow(attention, cmap='viridis')
        axes[0].set_xlabel("Image Patches")
        axes[0].set_ylabel("Text Tokens")
        axes[0].set_yticks(range(len(tokens)))
        axes[0].set_yticklabels(tokens)
        plt.colorbar(im, ax=axes[0], fraction=0.046)
        axes[0].set_title("Cross-Attention Matrix")
        
        # Image with highlighted regions
        # Average attention over text tokens
        avg_attention = attention.mean(axis=0)
        num_patches = int(np.sqrt(len(avg_attention)))
        attention_2d = avg_attention.reshape(num_patches, num_patches)
        attention_resized = cv2.resize(
            attention_2d,
            (image.shape[1], image.shape[0])
        )
        attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
        
        axes[1].imshow(image)
        axes[1].imshow(attention_resized, cmap='jet', alpha=0.4)
        axes[1].set_title("Image with Attention Overlay")
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig


class TokenAttribution:
    """
    Token-level attribution for text explanations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
    ):
        """
        Initialize token attribution.
        
        Args:
            model: VQA model
            tokenizer: Text tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def attribute(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Compute token attributions.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            pixel_values: Image tensor
            target_class: Target class for attribution
            **kwargs: Additional model kwargs
            
        Returns:
            Token attributions
        """
        # Get baseline (all zeros or padding)
        baseline_ids = torch.zeros_like(input_ids)
        
        # Compute integrated gradients
        steps = 20
        attributions = torch.zeros_like(input_ids, dtype=torch.float)
        
        for step in range(steps):
            # Interpolate between baseline and input
            alpha = step / steps
            
            # Enable gradients for embeddings
            embeddings = self.model.base_model.get_input_embeddings()
            
            # Get interpolated embeddings
            baseline_emb = embeddings(baseline_ids)
            input_emb = embeddings(input_ids)
            interpolated_emb = baseline_emb + alpha * (input_emb - baseline_emb)
            interpolated_emb.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(
                inputs_embeds=interpolated_emb,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
            
            logits = outputs['logits']
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            
            # Backward
            if target_class is None:
                target_class = logits.argmax(dim=-1)
            
            target_logits = logits[:, target_class].sum()
            target_logits.backward()
            
            # Accumulate gradients
            if interpolated_emb.grad is not None:
                attributions += interpolated_emb.grad.sum(dim=-1)
        
        # Normalize
        attributions = attributions / steps
        
        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'attributions': attributions.cpu().numpy(),
            'tokens': tokens,
        }
    
    def highlight_text(
        self,
        tokens: List[str],
        attributions: np.ndarray,
        threshold: float = 0.5,
    ) -> str:
        """
        Create highlighted text visualization.
        
        Args:
            tokens: List of tokens
            attributions: Attribution values
            threshold: Threshold for highlighting
            
        Returns:
            HTML string with highlighted tokens
        """
        # Normalize attributions
        attributions = attributions.flatten()
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
        
        html_parts = []
        
        for token, attr in zip(tokens, attributions):
            if attr > threshold:
                # High importance - red
                color = f"rgba(255, 0, 0, {attr:.2f})"
            else:
                # Low importance - blue
                color = f"rgba(0, 0, 255, {attr:.2f})"
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 3px;">{token}</span>'
            )
        
        return ' '.join(html_parts)


def visualize_attention_grid(
    attention_maps: List[np.ndarray],
    layer_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create grid visualization of attention maps.
    
    Args:
        attention_maps: List of attention matrices
        layer_names: Names for each layer
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_layers = len(attention_maps)
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for idx, attn in enumerate(attention_maps):
        row = idx // cols
        col = idx % cols
        
        ax = axes[row, col]
        im = ax.imshow(attn, cmap='viridis')
        
        title = layer_names[idx] if layer_names else f"Layer {idx + 1}"
        ax.set_title(title)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_layers, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test attention visualization
    print("Testing Attention Visualization...")
    
    # Create dummy attention map
    attention = np.random.rand(1, 196)
    
    # Create dummy image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test rollout visualization
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def forward(self, **kwargs):
            return {'logits': torch.randn(1, 100)}
    
    rollout = AttentionRollout(DummyModel())
    result = rollout.visualize(image, attention[0])
    
    print(f"Visualization shape: {result.shape}")
    print("Attention visualization test passed!")
