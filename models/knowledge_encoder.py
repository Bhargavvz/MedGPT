"""
Knowledge Encoder Module
========================
BioBERT/PubMedBERT encoder for medical knowledge integration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
from loguru import logger


class KnowledgeEncoder(nn.Module):
    """
    Knowledge encoder using BioBERT or PubMedBERT.
    
    Encodes medical knowledge snippets for integration with VQA model.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        hidden_size: int = 768,
        target_hidden_size: int = 1024,
        max_length: int = 256,
        freeze: bool = False,
        use_pooler: bool = True,
        pooling_strategy: str = "mean",  # mean, cls, max
    ):
        """
        Initialize knowledge encoder.
        
        Args:
            model_name: HuggingFace model name
            hidden_size: Encoder hidden size
            target_hidden_size: Target hidden size (for projection)
            max_length: Maximum token length
            freeze: Whether to freeze encoder
            use_pooler: Use pooler output
            pooling_strategy: How to pool token embeddings
        """
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.target_hidden_size = target_hidden_size
        self.max_length = max_length
        self.use_pooler = use_pooler
        self.pooling_strategy = pooling_strategy
        
        # Load encoder and tokenizer
        logger.info(f"Loading knowledge encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Update hidden size from config
        self.hidden_size = self.encoder.config.hidden_size
        
        # Projection layer to match target hidden size
        if self.hidden_size != target_hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, target_hidden_size),
                nn.LayerNorm(target_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
        
        # Freeze if specified
        if freeze:
            self._freeze()
        
        logger.info(f"Knowledge encoder initialized: {self.hidden_size} -> {target_hidden_size}")
    
    def _freeze(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Knowledge encoder frozen")
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt"
    ) -> Dict:
        """
        Tokenize input texts.
        
        Args:
            texts: Input text or list of texts
            return_tensors: Return tensor format
            
        Returns:
            Tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors
        )
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states based on strategy.
        
        Args:
            hidden_states: Token hidden states [B, L, H]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Pooled representation [B, H]
        """
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0]
        
        elif self.pooling_strategy == "max":
            # Mask padded tokens
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
            return hidden_states.max(dim=1)[0]
        
        else:  # mean
            # Mask padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            texts: Raw texts (will be tokenized)
            
        Returns:
            Dictionary with:
                - last_hidden_state: Token embeddings [B, L, H]
                - pooled_output: Pooled representation [B, H]
        """
        # Tokenize if texts provided
        if texts is not None:
            device = next(self.parameters()).device
            tokenized = self.tokenize(texts)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
        
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # Pool hidden states
        if self.use_pooler and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = self._pool(last_hidden_state, attention_mask)
        
        # Project to target hidden size
        last_hidden_state = self.projection(last_hidden_state)
        pooled_output = self.projection(pooled_output)
        
        return {
            'last_hidden_state': last_hidden_state,
            'pooled_output': pooled_output,
            'attention_mask': attention_mask
        }
    
    def encode(
        self,
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Input texts
            
        Returns:
            Text embeddings [B, H]
        """
        outputs = self.forward(texts=texts)
        return outputs['pooled_output']


class ConceptEncoder(nn.Module):
    """
    Encodes individual medical concepts from UMLS.
    """
    
    def __init__(
        self,
        base_encoder: KnowledgeEncoder,
        concept_embedding_dim: int = 256,
        num_concept_layers: int = 2,
    ):
        """
        Initialize concept encoder.
        
        Args:
            base_encoder: Base knowledge encoder
            concept_embedding_dim: Concept embedding dimension
            num_concept_layers: Number of transformer layers for concepts
        """
        super().__init__()
        
        self.base_encoder = base_encoder
        self.hidden_size = base_encoder.target_hidden_size
        
        # Concept-specific processing
        self.concept_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_concept_layers
        )
        
        # Concept projector
        self.concept_projector = nn.Sequential(
            nn.Linear(self.hidden_size, concept_embedding_dim),
            nn.LayerNorm(concept_embedding_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        concepts: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode medical concepts.
        
        Args:
            concepts: List of concept dictionaries with 'name' and 'definition'
            
        Returns:
            Encoded concepts
        """
        # Format concept texts
        concept_texts = []
        for concept in concepts:
            text = f"{concept.get('name', '')}: {concept.get('definition', '')}"
            concept_texts.append(text)
        
        if not concept_texts:
            # Return empty tensors
            device = next(self.parameters()).device
            return {
                'concept_embeddings': torch.zeros(1, 0, self.hidden_size, device=device),
                'pooled_concepts': torch.zeros(1, self.hidden_size, device=device)
            }
        
        # Encode with base encoder
        outputs = self.base_encoder(texts=concept_texts)
        
        # Process through concept transformer
        concept_hidden = outputs['last_hidden_state']
        concept_hidden = self.concept_transformer(concept_hidden)
        
        # Pool concepts
        pooled_concepts = self.concept_projector(outputs['pooled_output'])
        
        return {
            'concept_embeddings': concept_hidden,
            'pooled_concepts': pooled_concepts
        }


class HierarchicalKnowledgeEncoder(nn.Module):
    """
    Hierarchical knowledge encoder for multi-level medical knowledge.
    
    Encodes:
    1. Token-level: Fine-grained knowledge representations
    2. Sentence-level: Concept definitions
    3. Document-level: Overall knowledge context
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        target_hidden_size: int = 1024,
        num_aggregation_layers: int = 2,
    ):
        """
        Initialize hierarchical encoder.
        
        Args:
            model_name: Base model name
            target_hidden_size: Target hidden size
            num_aggregation_layers: Layers for knowledge aggregation
        """
        super().__init__()
        
        # Base encoder
        self.base_encoder = KnowledgeEncoder(
            model_name=model_name,
            target_hidden_size=target_hidden_size
        )
        
        self.hidden_size = target_hidden_size
        
        # Sentence-level aggregation
        self.sentence_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=target_hidden_size,
                nhead=8,
                dim_feedforward=target_hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_aggregation_layers
        )
        
        # Document-level aggregation
        self.document_aggregator = nn.Sequential(
            nn.Linear(target_hidden_size, target_hidden_size),
            nn.LayerNorm(target_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Level indicators
        self.level_embeddings = nn.Embedding(3, target_hidden_size)  # token, sentence, document
    
    def forward(
        self,
        knowledge_texts: List[str],
        return_all_levels: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode knowledge hierarchically.
        
        Args:
            knowledge_texts: List of knowledge texts
            return_all_levels: Return all hierarchy levels
            
        Returns:
            Multi-level knowledge encodings
        """
        if not knowledge_texts:
            device = next(self.parameters()).device
            empty = torch.zeros(1, self.hidden_size, device=device)
            return {
                'token_level': empty.unsqueeze(1),
                'sentence_level': empty.unsqueeze(1),
                'document_level': empty
            }
        
        # Token-level encoding
        base_outputs = self.base_encoder(texts=knowledge_texts)
        token_level = base_outputs['last_hidden_state']  # [B, L, H]
        
        # Sentence-level aggregation
        sentence_level = self.sentence_aggregator(token_level)  # [B, L, H]
        sentence_pooled = sentence_level.mean(dim=1, keepdim=True)  # [B, 1, H]
        
        # Document-level aggregation
        document_level = self.document_aggregator(sentence_pooled.squeeze(1))  # [B, H]
        
        result = {
            'document_level': document_level
        }
        
        if return_all_levels:
            result['token_level'] = token_level
            result['sentence_level'] = sentence_level
        
        return result


if __name__ == "__main__":
    # Test knowledge encoder
    print("Testing Knowledge Encoder...")
    
    # Create encoder (use smaller model for testing)
    try:
        encoder = KnowledgeEncoder(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            target_hidden_size=1024
        )
        
        # Test encoding
        texts = [
            "Pneumonia: Inflammation of the lung parenchyma",
            "Consolidation: Replacement of alveolar air by fluid"
        ]
        
        outputs = encoder(texts=texts)
        print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
        print(f"Pooled output shape: {outputs['pooled_output'].shape}")
        print("Knowledge encoder test passed!")
        
    except Exception as e:
        print(f"Could not test with PubMedBERT: {e}")
        print("Using bert-base for testing...")
        
        encoder = KnowledgeEncoder(
            model_name="bert-base-uncased",
            target_hidden_size=1024
        )
        
        outputs = encoder(texts=["test knowledge"])
        print(f"Fallback test - Pooled shape: {outputs['pooled_output'].shape}")
