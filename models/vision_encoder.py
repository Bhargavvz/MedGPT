"""
Vision Encoder Module
=====================
Wrapper for vision encoders (ViT, CLIP) for medical image encoding.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoImageProcessor,
)
from loguru import logger


class VisionEncoder(nn.Module):
    """
    Vision encoder wrapper for medical image feature extraction.
    
    Supports CLIP ViT-L/14, ViT models, and custom vision encoders.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        hidden_size: int = 1024,
        freeze: bool = True,
        gradient_checkpointing: bool = False,
        output_hidden_states: bool = True,
    ):
        """
        Initialize vision encoder.
        
        Args:
            model_name: HuggingFace model name
            hidden_size: Expected hidden size (for projection)
            freeze: Whether to freeze encoder weights
            gradient_checkpointing: Enable gradient checkpointing
            output_hidden_states: Return intermediate hidden states
        """
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.freeze = freeze
        self.output_hidden_states = output_hidden_states
        
        # Load model
        logger.info(f"Loading vision encoder: {model_name}")
        
        if "clip" in model_name.lower():
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.encoder_hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.encoder_hidden_size = self.encoder.config.hidden_size
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        
        # Projection layer if hidden sizes don't match
        if self.encoder_hidden_size != hidden_size:
            self.projection = nn.Linear(self.encoder_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if specified
        if freeze:
            self._freeze()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"Vision encoder initialized: {self.encoder_hidden_size} -> {hidden_size}")
    
    def _freeze(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen")
    
    def unfreeze(self, num_layers: Optional[int] = None):
        """
        Unfreeze encoder parameters.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            logger.info("Vision encoder fully unfrozen")
        else:
            # Unfreeze last N layers
            layers = list(self.encoder.modules())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Unfrozen last {num_layers} layers")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            return_dict: Return dictionary format
            
        Returns:
            Dictionary with:
                - last_hidden_state: Final hidden states [B, num_patches, hidden_size]
                - pooler_output: Pooled output [B, hidden_size]
                - hidden_states: All hidden states (if output_hidden_states=True)
        """
        # Get encoder outputs
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=self.output_hidden_states,
            return_dict=True
        )
        
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Project to target hidden size
        last_hidden_state = self.projection(last_hidden_state)
        last_hidden_state = self.layer_norm(last_hidden_state)
        
        # Get pooled output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooler_output = outputs.pooler_output
        else:
            pooler_output = last_hidden_state[:, 0]
        
        pooler_output = self.projection(pooler_output) if hasattr(self, 'projection') else pooler_output
        
        result = {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output,
        }
        
        if self.output_hidden_states and hasattr(outputs, 'hidden_states'):
            result['hidden_states'] = outputs.hidden_states
        
        return result
    
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get image features for downstream tasks.
        
        Args:
            pixel_values: Input images
            
        Returns:
            Image features [B, hidden_size]
        """
        outputs = self.forward(pixel_values)
        return outputs['pooler_output']
    
    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get patch-level embeddings.
        
        Args:
            pixel_values: Input images
            
        Returns:
            Patch embeddings [B, num_patches, hidden_size]
        """
        outputs = self.forward(pixel_values)
        return outputs['last_hidden_state']


class MedicalVisionEncoder(VisionEncoder):
    """
    Medical-specific vision encoder with additional features.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        hidden_size: int = 1024,
        num_medical_tokens: int = 32,
        use_medical_adapter: bool = True,
        **kwargs
    ):
        """
        Initialize medical vision encoder.
        
        Args:
            model_name: Base model name
            hidden_size: Hidden size
            num_medical_tokens: Number of learnable medical tokens
            use_medical_adapter: Use medical adapter layers
            **kwargs: Additional arguments for base class
        """
        super().__init__(model_name, hidden_size, **kwargs)
        
        self.use_medical_adapter = use_medical_adapter
        
        # Learnable medical tokens
        self.medical_tokens = nn.Parameter(
            torch.randn(1, num_medical_tokens, hidden_size) * 0.02
        )
        
        # Medical adapter (lightweight fine-tuning)
        if use_medical_adapter:
            self.medical_adapter = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, hidden_size),
                nn.Dropout(0.1)
            )
        
        logger.info(f"Medical vision encoder initialized with {num_medical_tokens} medical tokens")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with medical tokens."""
        # Get base encoder output
        outputs = super().forward(pixel_values, return_dict=True)
        
        last_hidden_state = outputs['last_hidden_state']
        batch_size = last_hidden_state.shape[0]
        
        # Expand medical tokens for batch
        medical_tokens = self.medical_tokens.expand(batch_size, -1, -1)
        
        # Concatenate medical tokens
        last_hidden_state = torch.cat([last_hidden_state, medical_tokens], dim=1)
        
        # Apply medical adapter
        if self.use_medical_adapter:
            adapter_output = self.medical_adapter(last_hidden_state)
            last_hidden_state = last_hidden_state + adapter_output
        
        outputs['last_hidden_state'] = last_hidden_state
        outputs['medical_tokens'] = medical_tokens
        
        return outputs


class ModalityAwareEncoder(nn.Module):
    """
    Modality-aware vision encoder with modality-specific adapters.
    """
    
    MODALITIES = ["CT", "MRI", "X-ray", "Ultrasound", "Pathology"]
    
    def __init__(
        self,
        base_encoder: VisionEncoder,
        hidden_size: int = 1024,
        adapter_hidden_size: int = 256,
    ):
        """
        Initialize modality-aware encoder.
        
        Args:
            base_encoder: Base vision encoder
            hidden_size: Hidden size
            adapter_hidden_size: Adapter hidden size
        """
        super().__init__()
        
        self.base_encoder = base_encoder
        self.hidden_size = hidden_size
        
        # Modality embedding
        self.modality_embedding = nn.Embedding(len(self.MODALITIES), hidden_size)
        
        # Modality-specific adapters
        self.modality_adapters = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(hidden_size, adapter_hidden_size),
                nn.GELU(),
                nn.Linear(adapter_hidden_size, hidden_size)
            )
            for modality in self.MODALITIES
        })
        
        self.modality_to_idx = {m: i for i, m in enumerate(self.MODALITIES)}
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        modality: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with modality awareness.
        
        Args:
            pixel_values: Input images
            modality: Modality string or list of modalities
            
        Returns:
            Encoded features
        """
        # Get base features
        outputs = self.base_encoder(pixel_values)
        features = outputs['last_hidden_state']
        
        if modality is not None:
            batch_size = features.shape[0]
            
            # Handle string or list input
            if isinstance(modality, str):
                modality = [modality] * batch_size
            
            # Apply modality-specific adapters
            for i, mod in enumerate(modality):
                if mod in self.modality_adapters:
                    adapter_output = self.modality_adapters[mod](features[i])
                    features[i] = features[i] + adapter_output
            
            # Add modality embeddings
            modality_indices = torch.tensor(
                [self.modality_to_idx.get(m, 0) for m in modality],
                device=features.device
            )
            modality_emb = self.modality_embedding(modality_indices)
            outputs['modality_embedding'] = modality_emb
        
        outputs['last_hidden_state'] = features
        return outputs


if __name__ == "__main__":
    # Test vision encoder
    print("Testing Vision Encoder...")
    
    # Create encoder
    encoder = VisionEncoder(
        model_name="openai/clip-vit-large-patch14",
        hidden_size=1024,
        freeze=True
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = encoder(dummy_input)
    
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Pooler output shape: {outputs['pooler_output'].shape}")
    print("Vision encoder test passed!")
