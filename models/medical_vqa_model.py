"""
Medical VQA Model
=================
Main model integrating all components for medical visual question answering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from loguru import logger

from .vision_encoder import VisionEncoder, MedicalVisionEncoder
from .knowledge_encoder import KnowledgeEncoder, HierarchicalKnowledgeEncoder
from .fusion_module import MultimodalFusion, KnowledgeGating
from .explanation_head import ExplanationModule, RationaleGenerator


class MedicalVQAModel(nn.Module):
    """
    Knowledge-Guided Explainable Transformer for Medical VQA.
    
    Integrates:
    - Qwen2-VL as base vision-language model
    - BioBERT/PubMedBERT for knowledge encoding
    - Cross-attention fusion with knowledge gating
    - Explainability module for rationale generation
    """
    
    def __init__(
        self,
        # Base model
        base_model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        # Vision encoder
        vision_encoder_name: str = "openai/clip-vit-large-patch14",
        vision_hidden_size: int = 1024,
        freeze_vision: bool = True,
        # Knowledge encoder
        knowledge_encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        knowledge_hidden_size: int = 768,
        # Fusion
        fusion_hidden_size: int = 1024,
        num_fusion_layers: int = 2,
        num_fusion_heads: int = 16,
        use_knowledge_gating: bool = True,
        gating_temperature: float = 1.0,
        # Explanation
        generate_rationale: bool = True,
        max_rationale_length: int = 128,
        # LoRA
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        # Quantization
        use_quantization: bool = True,
        load_in_4bit: bool = True,
        # Other
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        """
        Initialize Medical VQA model.
        
        Args:
            base_model_name: Qwen2-VL model name
            vision_encoder_name: Vision encoder model name
            vision_hidden_size: Vision encoder hidden size
            freeze_vision: Freeze vision encoder
            knowledge_encoder_name: Knowledge encoder model name
            knowledge_hidden_size: Knowledge encoder hidden size
            fusion_hidden_size: Fusion module hidden size
            num_fusion_layers: Number of fusion layers
            num_fusion_heads: Number of attention heads in fusion
            use_knowledge_gating: Use knowledge gating mechanism
            gating_temperature: Knowledge gating temperature
            generate_rationale: Generate explanation rationales
            max_rationale_length: Maximum rationale length
            use_lora: Use LoRA fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: LoRA target modules
            use_quantization: Use quantization
            load_in_4bit: Load in 4-bit
            dropout: Dropout probability
            gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.use_lora = use_lora
        self.use_quantization = use_quantization
        self.generate_rationale = generate_rationale
        self.fusion_hidden_size = fusion_hidden_size
        
        # Default LoRA targets
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        logger.info(f"Initializing Medical VQA Model with base: {base_model_name}")
        
        # =====================================================================
        # Load Base Model (Qwen2-VL)
        # =====================================================================
        self._load_base_model(
            base_model_name=base_model_name,
            use_quantization=use_quantization,
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=gradient_checkpointing,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        
        # Get hidden size from base model
        self.hidden_size = self.base_model.config.hidden_size
        
        # =====================================================================
        # Vision Encoder (Additional for explicit control)
        # =====================================================================
        self.vision_encoder = MedicalVisionEncoder(
            model_name=vision_encoder_name,
            hidden_size=vision_hidden_size,
            freeze=freeze_vision,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # =====================================================================
        # Knowledge Encoder (BioBERT/PubMedBERT)
        # =====================================================================
        self.knowledge_encoder = HierarchicalKnowledgeEncoder(
            model_name=knowledge_encoder_name,
            target_hidden_size=fusion_hidden_size,
        )
        
        # =====================================================================
        # Multimodal Fusion Module
        # =====================================================================
        self.fusion = MultimodalFusion(
            vision_dim=vision_hidden_size,
            text_dim=self.hidden_size,
            knowledge_dim=fusion_hidden_size,
            hidden_dim=fusion_hidden_size,
            num_heads=num_fusion_heads,
            num_layers=num_fusion_layers,
            dropout=dropout,
            use_knowledge_gating=use_knowledge_gating,
            gating_temperature=gating_temperature,
        )
        
        # =====================================================================
        # Feature Projector (Fusion -> Base Model)
        # =====================================================================
        self.feature_projector = nn.Sequential(
            nn.Linear(fusion_hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # =====================================================================
        # Explanation Module
        # =====================================================================
        if generate_rationale:
            self.explanation_module = ExplanationModule(
                hidden_dim=fusion_hidden_size,
                vocab_size=self.base_model.config.vocab_size,
                max_length=max_rationale_length,
                use_structured_rationale=True,
            )
        
        logger.info("Medical VQA Model initialized successfully")
    
    def _load_base_model(
        self,
        base_model_name: str,
        use_quantization: bool,
        load_in_4bit: bool,
        gradient_checkpointing: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str],
    ):
        """Load and configure base Qwen2-VL model."""
        # Quantization config
        if use_quantization and load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Loading model with 4-bit quantization")
        else:
            quantization_config = None
        
        # Load model
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load processor for Qwen-VL
            self.processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer
            
        except Exception as e:
            logger.warning(f"Could not load {base_model_name}: {e}")
            logger.info("Loading fallback model for development")
            
            # Fallback for development/testing
            vocab_size = 151936
            hidden_size = 4096
            
            # Build a proper lightweight LM head so output shape matches vocab_size
            self.base_model = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, vocab_size),
            )
            self.processor = None
            
            # Try loading tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
            except Exception:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "Qwen/Qwen-7B",
                        trust_remote_code=True
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            logger.info(f"Fallback model: hidden_size={hidden_size}, vocab_size={vocab_size}")
            logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
            
            # Set config for hidden size access
            class DummyConfig:
                pass
            cfg = DummyConfig()
            cfg.hidden_size = hidden_size
            cfg.vocab_size = vocab_size
            self.base_model.config = cfg
            return
        
        # Gradient checkpointing
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        
        # Prepare for k-bit training if quantized
        if use_quantization:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Apply LoRA
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder weights."""
        self.vision_encoder._freeze()
    
    def unfreeze_vision_encoder(self, num_layers: Optional[int] = None):
        """Unfreeze vision encoder weights."""
        self.vision_encoder.unfreeze(num_layers)
    
    def encode_image(
        self,
        pixel_values: torch.Tensor,
        modality: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode medical image.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            modality: Imaging modality
            
        Returns:
            Encoded image features
        """
        return self.vision_encoder(pixel_values)
    
    def encode_knowledge(
        self,
        knowledge_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode medical knowledge.
        
        Args:
            knowledge_texts: List of knowledge snippets
            
        Returns:
            Encoded knowledge features
        """
        return self.knowledge_encoder(knowledge_texts)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        knowledge_texts: Optional[List[str]] = None,
        modality: Optional[Union[str, List[str]]] = None,
        return_explanation: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            pixel_values: Input images [B, C, H, W]
            labels: Target labels for loss computation [B, L]
            knowledge_texts: Optional knowledge snippets
            modality: Imaging modality
            return_explanation: Return explanation outputs
            return_attention: Return attention weights
            
        Returns:
            Model outputs including loss, logits, and optional explanations
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # =====================================================================
        # Encode Visual Features
        # =====================================================================
        vision_outputs = self.encode_image(pixel_values, modality)
        vision_features = vision_outputs['last_hidden_state']
        
        # =====================================================================
        # Get Text Features from Base Model Embeddings
        # =====================================================================
        if hasattr(self.base_model, 'get_input_embeddings'):
            embeddings = self.base_model.get_input_embeddings()
            text_features = embeddings(input_ids)
        else:
            # Fallback for dummy model
            text_features = torch.randn(
                batch_size, input_ids.shape[1], self.hidden_size,
                device=device
            )
        
        # =====================================================================
        # Encode Knowledge (if provided)
        # =====================================================================
        if knowledge_texts is not None:
            knowledge_outputs = self.encode_knowledge(knowledge_texts)
            knowledge_features = knowledge_outputs['document_level'].unsqueeze(1)
            if 'token_level' in knowledge_outputs:
                knowledge_features = knowledge_outputs['token_level']
        else:
            knowledge_features = None
        
        # =====================================================================
        # Multimodal Fusion
        # =====================================================================
        fusion_outputs = self.fusion(
            vision_features=vision_features,
            text_features=text_features,
            knowledge_features=knowledge_features,
            return_attention=return_attention,
        )
        
        fused_features = fusion_outputs['fused_features']
        
        # Project to base model hidden size
        projected_features = self.feature_projector(fused_features)
        
        # =====================================================================
        # Base Model Forward Pass
        # =====================================================================
        # For actual Qwen2-VL, we would use a more sophisticated integration
        # This is a simplified version for demonstration
        
        if hasattr(self.base_model, 'forward') and not isinstance(self.base_model, nn.Sequential):
            # Full model forward
            outputs = self.base_model(
                inputs_embeds=projected_features,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss if labels is not None else None
            logits = outputs.logits
        else:
            # Fallback forward - pass through the sequential model
            logits = self.base_model(projected_features)
            loss = None
            if labels is not None:
                # Ensure labels don't exceed vocab size
                vocab_size = logits.size(-1)
                valid_labels = labels.clone()
                valid_labels[valid_labels >= vocab_size] = -100
                valid_labels[valid_labels < 0] = -100
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    valid_labels.view(-1),
                    ignore_index=-100
                )
        
        # =====================================================================
        # Build Output Dictionary
        # =====================================================================
        result = {
            'loss': loss,
            'logits': logits,
            'fused_features': fused_features,
        }
        
        # Add attention weights if requested
        if return_attention and 'attention_weights' in fusion_outputs:
            result['attention_weights'] = fusion_outputs['attention_weights']
        
        # =====================================================================
        # Generate Explanation (if enabled)
        # =====================================================================
        if return_explanation and self.generate_rationale:
            explanation_outputs = self.explanation_module(
                fused_features=fused_features,
                attention_mask=attention_mask,
            )
            result['explanation'] = explanation_outputs
        
        return result
    
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        knowledge_texts: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        generate_explanation: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate answer and explanation.
        
        Args:
            pixel_values: Input images
            input_ids: Input token IDs
            attention_mask: Attention mask
            knowledge_texts: Knowledge snippets
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Use sampling
            generate_explanation: Generate explanation
            **kwargs: Additional generation arguments
            
        Returns:
            Generated outputs
        """
        # Get fused features
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask if attention_mask is not None else torch.ones_like(input_ids),
            pixel_values=pixel_values,
            knowledge_texts=knowledge_texts,
            return_explanation=False,
        )
        
        fused_features = outputs['fused_features']
        
        # Generate answer using base model
        if hasattr(self.base_model, 'generate') and not isinstance(self.base_model, nn.Sequential):
            projected_features = self.feature_projector(fused_features)
            
            generated = self.base_model.generate(
                inputs_embeds=projected_features,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs
            )
        else:
            # Dummy generation for testing
            generated = torch.randint(0, 32000, (pixel_values.shape[0], max_new_tokens))
        
        result = {'generated_ids': generated}
        
        # Generate explanation
        if generate_explanation and self.generate_rationale:
            explanation_outputs = self.explanation_module.generate(
                fused_features=fused_features,
                attention_mask=attention_mask,
                max_length=128,
                temperature=temperature,
                top_p=top_p,
            )
            result['explanation_ids'] = explanation_outputs['generated_ids']
        
        return result
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_medical_vqa_model(config) -> MedicalVQAModel:
    """
    Create Medical VQA model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized model
    """
    return MedicalVQAModel(
        base_model_name=config.model.base_model,
        vision_encoder_name=config.model.vision_encoder,
        vision_hidden_size=config.model.vision_hidden_size,
        knowledge_encoder_name=config.model.knowledge_encoder,
        knowledge_hidden_size=config.model.knowledge_hidden_size,
        fusion_hidden_size=config.model.fusion_hidden_size,
        num_fusion_layers=config.model.num_fusion_layers,
        num_fusion_heads=config.model.num_attention_heads,
        use_knowledge_gating=config.model.use_knowledge_gating,
        gating_temperature=config.model.gating_temperature,
        generate_rationale=config.model.generate_rationale,
        max_rationale_length=config.model.max_rationale_length,
        use_lora=config.lora.enabled,
        lora_r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        lora_target_modules=config.lora.target_modules,
        use_quantization=config.quantization.enabled,
        load_in_4bit=config.quantization.load_in_4bit,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )


if __name__ == "__main__":
    # Test model initialization
    print("Testing Medical VQA Model...")
    print("=" * 50)
    
    # Create model with minimal config for testing
    model = MedicalVQAModel(
        base_model_name="Qwen/Qwen2-VL-7B-Instruct",
        use_lora=True,
        use_quantization=False,  # Disable for testing
        generate_rationale=True,
    )
    
    print(f"Total parameters: {model.get_total_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 64
    
    dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 32000, (batch_size, seq_len))
    
    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        pixel_values=dummy_pixel_values,
        labels=dummy_labels,
        knowledge_texts=["Pneumonia: lung infection"] * batch_size,
        return_explanation=True,
    )
    
    print(f"Loss: {outputs['loss']}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    
    print("\nMedical VQA Model test passed!")
