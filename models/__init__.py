"""Models module initialization."""
from .vision_encoder import VisionEncoder, MedicalVisionEncoder, ModalityAwareEncoder
from .knowledge_encoder import KnowledgeEncoder, ConceptEncoder, HierarchicalKnowledgeEncoder
from .fusion_module import (
    CrossAttention,
    CrossAttentionBlock,
    KnowledgeGating,
    MultimodalFusion,
    GatedFusion,
)
from .explanation_head import (
    ExplanationHead,
    RationaleGenerator,
    ExplanationModule,
)
from .medical_vqa_model import MedicalVQAModel, create_medical_vqa_model

__all__ = [
    # Vision
    "VisionEncoder",
    "MedicalVisionEncoder",
    "ModalityAwareEncoder",
    # Knowledge
    "KnowledgeEncoder",
    "ConceptEncoder",
    "HierarchicalKnowledgeEncoder",
    # Fusion
    "CrossAttention",
    "CrossAttentionBlock",
    "KnowledgeGating",
    "MultimodalFusion",
    "GatedFusion",
    # Explanation
    "ExplanationHead",
    "RationaleGenerator",
    "ExplanationModule",
    # Main Model
    "MedicalVQAModel",
    "create_medical_vqa_model",
]
