"""
Grad-CAM Module
===============
Gradient-weighted Class Activation Mapping for visual explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import cv2
from loguru import logger


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Produces visual explanations for predictions by highlighting
    important regions in the input image.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_layer_name: Optional[str] = None,
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The model to explain
            target_layer: Target layer for CAM (usually last conv layer)
            target_layer_name: Name of target layer (alternative to target_layer)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif target_layer_name is not None:
            self.target_layer = self._find_layer_by_name(target_layer_name)
        else:
            self.target_layer = self._find_last_conv_layer()
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_layer_by_name(self, name: str) -> nn.Module:
        """Find layer by name in model."""
        for n, module in self.model.named_modules():
            if n == name:
                return module
        raise ValueError(f"Layer {name} not found in model")
    
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            # Try to find in vision encoder
            if hasattr(self.model, 'vision_encoder'):
                for module in self.model.vision_encoder.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
        
        if last_conv is None:
            raise ValueError("Could not find convolutional layer for Grad-CAM")
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **model_kwargs
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [B, C, H, W]
            target_class: Target class index (None = use predicted class)
            **model_kwargs: Additional model forward kwargs
            
        Returns:
            Heatmap array [B, H, W]
        """
        # Forward pass
        self.model.zero_grad()
        
        if 'pixel_values' not in model_kwargs:
            model_kwargs['pixel_values'] = input_tensor
        
        output = self.model(**model_kwargs)
        
        # Get logits
        if isinstance(output, dict):
            logits = output.get('logits', output.get('output'))
        else:
            logits = output
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        
        # Backward pass
        if logits.dim() == 3:  # [B, L, V]
            target_logits = logits[:, -1, :]  # Use last token
        else:
            target_logits = logits
        
        one_hot = torch.zeros_like(target_logits)
        if isinstance(target_class, int):
            one_hot[:, target_class] = 1
        else:
            for i, c in enumerate(target_class):
                one_hot[i, c] = 1
        
        target_logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            logger.warning("No gradients or activations captured")
            return np.zeros((input_tensor.shape[0], 14, 14))
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=[2, 3] if gradients.dim() == 4 else [-1], keepdim=True)
        
        # Weighted combination of activation maps
        if activations.dim() == 4:
            cam = (weights * activations).sum(dim=1)
        else:
            # Handle transformer-style outputs
            cam = (weights * activations).sum(dim=-1)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = self._normalize(cam)
        
        return cam.detach().cpu().numpy()
    
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        """Normalize CAM to [0, 1]."""
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
    
    def visualize(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image (H, W, 3) uint8
            heatmap: CAM heatmap (H', W')
            alpha: Transparency for overlay
            colormap: OpenCV colormap
            
        Returns:
            Visualization image (H, W, 3)
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(
            heatmap,
            (image.shape[1], image.shape[0])
        )
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )
        
        # Overlay
        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Grad-CAM with weighted gradients.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **model_kwargs
    ) -> np.ndarray:
        """Compute Grad-CAM++ heatmap."""
        # Forward pass
        self.model.zero_grad()
        
        if 'pixel_values' not in model_kwargs:
            model_kwargs['pixel_values'] = input_tensor
        
        output = self.model(**model_kwargs)
        
        if isinstance(output, dict):
            logits = output.get('logits', output.get('output'))
        else:
            logits = output
        
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        
        # Backward pass
        if logits.dim() == 3:
            target_logits = logits[:, -1, :]
        else:
            target_logits = logits
        
        one_hot = torch.zeros_like(target_logits)
        if isinstance(target_class, int):
            one_hot[:, target_class] = 1
        else:
            for i, c in enumerate(target_class):
                one_hot[i, c] = 1
        
        target_logits.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return np.zeros((input_tensor.shape[0], 14, 14))
        
        # Grad-CAM++ weights
        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3
        
        # Sum of activations
        sum_activations = activations.sum(dim=[2, 3] if activations.dim() == 4 else [-1], keepdim=True)
        
        # Alpha coefficients
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = alpha_num / alpha_denom
        
        # Weights
        weights = (alpha * F.relu(gradients)).sum(dim=[2, 3] if gradients.dim() == 4 else [-1], keepdim=True)
        
        # CAM
        if activations.dim() == 4:
            cam = (weights * activations).sum(dim=1)
        else:
            cam = (weights * activations).sum(dim=-1)
        
        cam = F.relu(cam)
        cam = self._normalize(cam)
        
        return cam.detach().cpu().numpy()


class ScoreCAM(GradCAM):
    """
    Score-CAM: Gradient-free CAM using activation scores.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        **model_kwargs
    ) -> np.ndarray:
        """Compute Score-CAM heatmap."""
        with torch.no_grad():
            # Get activations
            if 'pixel_values' not in model_kwargs:
                model_kwargs['pixel_values'] = input_tensor
            
            _ = self.model(**model_kwargs)
            activations = self.activations
            
            if activations is None:
                return np.zeros((input_tensor.shape[0], 14, 14))
            
            # Get baseline prediction
            output = self.model(**model_kwargs)
            if isinstance(output, dict):
                logits = output.get('logits', output.get('output'))
            else:
                logits = output
            
            if logits.dim() == 3:
                baseline = logits[:, -1, :]
            else:
                baseline = logits
            
            if target_class is None:
                target_class = baseline.argmax(dim=-1)
            
            # Process each activation map
            batch_size = input_tensor.shape[0]
            
            if activations.dim() == 4:
                num_activations = activations.shape[1]
                h, w = activations.shape[2], activations.shape[3]
            else:
                # Handle ViT-style outputs
                num_activations = activations.shape[-1]
                h = w = int(np.sqrt(activations.shape[1]))
            
            # Limit for efficiency
            num_activations = min(num_activations, 256)
            
            scores = torch.zeros(batch_size, num_activations, device=input_tensor.device)
            
            for i in range(num_activations):
                # Create masked input
                if activations.dim() == 4:
                    mask = activations[:, i:i+1, :, :]
                else:
                    mask = activations[:, :, i:i+1]
                
                # Resize mask to input size
                if activations.dim() == 4:
                    mask_resized = F.interpolate(
                        mask,
                        size=input_tensor.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    mask = mask.view(batch_size, h, w, 1).permute(0, 3, 1, 2)
                    mask_resized = F.interpolate(
                        mask,
                        size=input_tensor.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Normalize mask
                mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)
                
                # Masked input
                masked_input = input_tensor * mask_norm
                
                # Get prediction
                model_kwargs['pixel_values'] = masked_input
                output = self.model(**model_kwargs)
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('output'))
                else:
                    logits = output
                
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                
                # Score
                softmax = F.softmax(logits, dim=-1)
                for b in range(batch_size):
                    tc = target_class if isinstance(target_class, int) else target_class[b].item()
                    scores[b, i] = softmax[b, tc]
            
            # Weighted combination
            scores = F.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)
            
            if activations.dim() == 4:
                cam = (scores * activations[:, :num_activations]).sum(dim=1)
            else:
                act_reshaped = activations[:, :, :num_activations].view(batch_size, h, w, num_activations).permute(0, 3, 1, 2)
                cam = (scores * act_reshaped).sum(dim=1)
            
            cam = F.relu(cam)
            cam = self._normalize(cam)
            
            return cam.cpu().numpy()


class MedicalGradCAM:
    """
    Specialized Grad-CAM for Medical VQA.
    
    Provides domain-specific visualizations and analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vision_encoder_layer: Optional[str] = None,
    ):
        """
        Initialize Medical Grad-CAM.
        
        Args:
            model: Medical VQA model
            vision_encoder_layer: Layer in vision encoder for CAM
        """
        self.model = model
        
        # Setup Grad-CAM for vision encoder
        if hasattr(model, 'vision_encoder'):
            self.grad_cam = GradCAM(
                model.vision_encoder,
                target_layer_name=vision_encoder_layer
            )
        else:
            self.grad_cam = GradCAM(model)
    
    def explain(
        self,
        image: torch.Tensor,
        question: str,
        predicted_answer: str,
        tokenizer=None,
        **kwargs
    ) -> Dict:
        """
        Generate visual explanation.
        
        Args:
            image: Input image tensor
            question: Question text
            predicted_answer: Model's predicted answer
            tokenizer: Tokenizer for encoding
            **kwargs: Additional model kwargs
            
        Returns:
            Dictionary with explanation components
        """
        # Get heatmap
        heatmap = self.grad_cam(image, **kwargs)
        
        # Find important regions
        important_regions = self._find_important_regions(heatmap)
        
        # Generate region description
        region_description = self._describe_regions(important_regions)
        
        return {
            'heatmap': heatmap,
            'important_regions': important_regions,
            'region_description': region_description,
            'question': question,
            'answer': predicted_answer,
        }
    
    def _find_important_regions(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """Find important regions in heatmap."""
        regions = []
        
        for i, h in enumerate(heatmap):
            # Threshold
            binary = (h > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                x, y, w, h_box = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                if area > 10:  # Filter small regions
                    regions.append({
                        'batch_idx': i,
                        'bbox': (x, y, w, h_box),
                        'area': area,
                        'mean_activation': float(h[y:y+h_box, x:x+w].mean()),
                    })
        
        return regions
    
    def _describe_regions(self, regions: List[Dict]) -> str:
        """Generate description of important regions."""
        if not regions:
            return "No significant regions detected."
        
        # Sort by activation
        regions = sorted(regions, key=lambda x: x['mean_activation'], reverse=True)
        
        descriptions = []
        for i, region in enumerate(regions[:3]):
            x, y, w, h = region['bbox']
            desc = f"Region {i+1}: Located at position ({x}, {y}) with activation {region['mean_activation']:.2f}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)


if __name__ == "__main__":
    # Test Grad-CAM
    print("Testing Grad-CAM...")
    
    # Create dummy model
    class DummyVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, 100)
        
        def forward(self, pixel_values):
            x = F.relu(self.conv1(pixel_values))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return {'logits': self.fc(x)}
    
    model = DummyVisionModel()
    cam = GradCAM(model, target_layer=model.conv2)
    
    # Test input
    dummy_input = torch.randn(2, 3, 224, 224)
    heatmap = cam(dummy_input)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print("Grad-CAM test passed!")
