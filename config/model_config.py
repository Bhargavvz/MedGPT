"""
Medical VQA Configuration Module
================================
Type-safe configuration using Python dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import yaml
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    base_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    vision_encoder: str = "openai/clip-vit-large-patch14"
    knowledge_encoder: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    hidden_size: int = 4096
    vision_hidden_size: int = 1024
    knowledge_hidden_size: int = 768
    fusion_hidden_size: int = 1024
    num_attention_heads: int = 16
    
    fusion_type: str = "cross_attention"
    num_fusion_layers: int = 2
    
    use_knowledge_gating: bool = True
    gating_temperature: float = 1.0
    
    generate_rationale: bool = True
    max_rationale_length: int = 128


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    enabled: bool = True
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """Quantization (QLoRA) configuration."""
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./checkpoints"
    num_epochs: int = 15
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    optimizer: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    
    fp16: bool = True
    bf16: bool = False
    
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    freeze_vision_encoder: bool = True
    unfreeze_vision_epoch: int = 3
    
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    
    logging_steps: int = 10
    logging_dir: str = "./logs"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    eval_strategy: str = "steps"
    eval_steps: int = 500
    
    resume_from_checkpoint: Optional[str] = None


@dataclass
class LossConfig:
    """Loss function weights."""
    answer_loss_weight: float = 1.0
    attention_alignment_weight: float = 0.1
    knowledge_grounding_weight: float = 0.2
    rationale_generation_weight: float = 0.3


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    horizontal_flip_prob: float = 0.5
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    gaussian_noise_var_limit: Tuple[int, int] = (10, 50)


@dataclass
class DataConfig:
    """Data configuration."""
    train_file: str = "./data/processed/train.json"
    val_file: str = "./data/processed/val.json"
    test_file: str = "./data/processed/test.json"
    
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    max_question_length: int = 128
    max_answer_length: int = 256
    max_knowledge_length: int = 256
    
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    use_augmentation: bool = True
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class KnowledgeConfig:
    """Knowledge retrieval configuration."""
    umls_api_key: Optional[str] = None
    use_scispacy: bool = True
    scispacy_model: str = "en_core_sci_lg"
    top_k_concepts: int = 5
    max_definition_length: int = 100
    sources: List[str] = field(default_factory=lambda: ["UMLS", "RadLex", "SNOMED-CT"])


@dataclass
class GradCAMConfig:
    """Grad-CAM configuration."""
    enabled: bool = True
    target_layer: str = "vision_encoder.encoder.layers[-1]"


@dataclass
class AttentionRolloutConfig:
    """Attention rollout configuration."""
    enabled: bool = True
    head_fusion: str = "mean"
    discard_ratio: float = 0.9


@dataclass
class IntegratedGradientsConfig:
    """Integrated gradients configuration."""
    enabled: bool = True
    n_steps: int = 50
    internal_batch_size: int = 16


@dataclass
class RationaleConfig:
    """Rationale generation configuration."""
    enabled: bool = True
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class ExplainabilityConfig:
    """Explainability configuration."""
    grad_cam: GradCAMConfig = field(default_factory=GradCAMConfig)
    attention_rollout: AttentionRolloutConfig = field(default_factory=AttentionRolloutConfig)
    integrated_gradients: IntegratedGradientsConfig = field(default_factory=IntegratedGradientsConfig)
    rationale: RationaleConfig = field(default_factory=RationaleConfig)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str = "./checkpoints/best_model"
    device: str = "cuda"
    batch_size: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    generate_heatmap: bool = True
    generate_rationale: bool = True


@dataclass
class WebAppConfig:
    """Web application configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 50
    allowed_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".dcm"])
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "bleu", "rouge_l", "f1", "exact_match"
    ])
    human_eval_enabled: bool = False
    human_eval_num_samples: int = 100


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""
    enabled: bool = True
    config_file: str = "./training/deepspeed_config.json"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    webapp: WebAppConfig = field(default_factory=WebAppConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Use OmegaConf for structured merging
        schema = OmegaConf.structured(cls)
        loaded = OmegaConf.create(yaml_config)
        merged = OmegaConf.merge(schema, loaded)
        
        return OmegaConf.to_object(merged)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = OmegaConf.to_container(OmegaConf.structured(self))
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration, optionally loading from file."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config()


# Convenience function to get default paths
def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_default_config_path() -> Path:
    """Get default config file path."""
    return get_project_root() / "config" / "config.yaml"
