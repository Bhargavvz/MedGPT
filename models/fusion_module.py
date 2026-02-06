"""
Fusion Module
=============
Cross-attention transformer for multimodal fusion with knowledge gating.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class CrossAttention(nn.Module):
    """
    Cross-attention module for fusing different modalities.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize cross-attention.
        
        Args:
            query_dim: Query dimension
            key_dim: Key/Value dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Use bias in projections
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(key_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(key_dim, hidden_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            query: Query tensor [B, Lq, Dq]
            key: Key tensor [B, Lk, Dk]
            value: Value tensor [B, Lk, Dk], defaults to key
            attention_mask: Attention mask [B, Lq, Lk]
            return_attention: Return attention weights
            
        Returns:
            Output tensor [B, Lq, H], optionally attention weights
        """
        if value is None:
            value = key
        
        batch_size = query.shape[0]
        
        # Project queries, keys, values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0,
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
        attn_output = self.o_proj(attn_output)
        
        if return_attention:
            return attn_output, attn_weights
        return attn_output


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        intermediate_dim = intermediate_dim or hidden_dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with feed-forward and residual connections.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pre_norm: bool = True,
    ):
        """
        Initialize cross-attention block.
        
        Args:
            query_dim: Query dimension
            key_dim: Key/Value dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_pre_norm: Use pre-layer normalization
        """
        super().__init__()
        
        self.use_pre_norm = use_pre_norm
        
        # Cross attention
        self.cross_attn = CrossAttention(
            query_dim=hidden_dim,
            key_dim=key_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward
        self.ff = FeedForward(hidden_dim, dropout=dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Query projection if dimensions differ
        if query_dim != hidden_dim:
            self.query_proj = nn.Linear(query_dim, hidden_dim)
        else:
            self.query_proj = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        # Project query
        query = self.query_proj(query)
        
        if self.use_pre_norm:
            # Pre-norm: Norm -> Attention -> Residual
            normed = self.norm1(query)
            attn_output = self.cross_attn(
                normed, key, value, attention_mask, return_attention
            )
            
            if return_attention:
                attn_output, attn_weights = attn_output
            
            query = query + self.dropout(attn_output)
            query = query + self.ff(self.norm2(query))
        else:
            # Post-norm: Attention -> Residual -> Norm
            attn_output = self.cross_attn(
                query, key, value, attention_mask, return_attention
            )
            
            if return_attention:
                attn_output, attn_weights = attn_output
            
            query = self.norm1(query + self.dropout(attn_output))
            query = self.norm2(query + self.ff(query))
        
        if return_attention:
            return query, attn_weights
        return query


class KnowledgeGating(nn.Module):
    """
    Knowledge gating mechanism to control knowledge integration.
    
    Adaptively gates knowledge based on query relevance.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        knowledge_dim: int,
        temperature: float = 1.0,
    ):
        """
        Initialize knowledge gating.
        
        Args:
            hidden_dim: Query hidden dimension
            knowledge_dim: Knowledge hidden dimension
            temperature: Softmax temperature
        """
        super().__init__()
        
        self.temperature = temperature
        
        # Gate computation
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim + knowledge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Knowledge projection
        if knowledge_dim != hidden_dim:
            self.knowledge_proj = nn.Linear(knowledge_dim, hidden_dim)
        else:
            self.knowledge_proj = nn.Identity()
    
    def forward(
        self,
        query_features: torch.Tensor,
        knowledge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply knowledge gating.
        
        Args:
            query_features: Query features [B, L, H]
            knowledge_features: Knowledge features [B, K, Hk]
            
        Returns:
            Gated knowledge, gate values
        """
        # Project knowledge
        knowledge = self.knowledge_proj(knowledge_features)
        
        # Compute relevance between query and knowledge
        # Average pool over sequence length
        query_pooled = query_features.mean(dim=1, keepdim=True)  # [B, 1, H]
        knowledge_pooled = knowledge.mean(dim=1, keepdim=True)  # [B, 1, H]
        
        # Concatenate for gate computation
        combined = torch.cat([query_pooled, knowledge_pooled], dim=-1)  # [B, 1, 2H]
        
        # Compute gate value
        gate = self.gate_proj(combined) / self.temperature  # [B, 1, 1]
        
        # Apply gate to knowledge
        gated_knowledge = knowledge * gate
        
        return gated_knowledge, gate.squeeze(-1)


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module combining vision, text, and knowledge.
    """
    
    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 4096,
        knowledge_dim: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_knowledge_gating: bool = True,
        gating_temperature: float = 1.0,
    ):
        """
        Initialize multimodal fusion.
        
        Args:
            vision_dim: Vision encoder hidden dimension
            text_dim: Text encoder hidden dimension
            knowledge_dim: Knowledge encoder hidden dimension
            hidden_dim: Fusion hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of fusion layers
            dropout: Dropout probability
            use_knowledge_gating: Use knowledge gating mechanism
            gating_temperature: Temperature for knowledge gating
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_knowledge_gating = use_knowledge_gating
        
        # Input projections
        self.vision_proj = nn.Linear(vision_dim, hidden_dim) if vision_dim != hidden_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()
        self.knowledge_proj = nn.Linear(knowledge_dim, hidden_dim) if knowledge_dim != hidden_dim else nn.Identity()
        
        # Knowledge gating
        if use_knowledge_gating:
            self.knowledge_gate = KnowledgeGating(
                hidden_dim=hidden_dim,
                knowledge_dim=hidden_dim,
                temperature=gating_temperature
            )
        
        # Vision-text fusion layers
        self.vision_text_fusion = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Knowledge integration layers
        self.knowledge_fusion = nn.ModuleList([
            CrossAttentionBlock(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        knowledge_features: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        knowledge_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal fusion.
        
        Args:
            vision_features: Vision encoder output [B, Lv, Dv]
            text_features: Text encoder output [B, Lt, Dt]
            knowledge_features: Knowledge encoder output [B, Lk, Dk]
            vision_mask: Vision attention mask
            text_mask: Text attention mask
            knowledge_mask: Knowledge attention mask
            return_attention: Return attention weights
            
        Returns:
            Dictionary with fused features and optional attention
        """
        attention_weights = {}
        
        # Project inputs
        vision = self.vision_proj(vision_features)
        text = self.text_proj(text_features)
        
        # Vision-text fusion: text attends to vision
        fused = text
        for i, layer in enumerate(self.vision_text_fusion):
            if return_attention:
                fused, attn = layer(fused, vision, attention_mask=vision_mask, return_attention=True)
                attention_weights[f'vision_text_layer_{i}'] = attn
            else:
                fused = layer(fused, vision, attention_mask=vision_mask)
        
        # Knowledge integration
        if knowledge_features is not None:
            knowledge = self.knowledge_proj(knowledge_features)
            
            # Apply knowledge gating
            if self.use_knowledge_gating:
                knowledge, gate_values = self.knowledge_gate(fused, knowledge)
                attention_weights['knowledge_gate'] = gate_values
            
            # Fuse with knowledge
            knowledge_fused = fused
            for i, layer in enumerate(self.knowledge_fusion):
                if return_attention:
                    knowledge_fused, attn = layer(
                        knowledge_fused, knowledge,
                        attention_mask=knowledge_mask,
                        return_attention=True
                    )
                    attention_weights[f'knowledge_layer_{i}'] = attn
                else:
                    knowledge_fused = layer(
                        knowledge_fused, knowledge,
                        attention_mask=knowledge_mask
                    )
            
            # Combine vision-text and knowledge branches
            fused = self.final_fusion(torch.cat([fused, knowledge_fused], dim=-1))
        
        # Output normalization
        fused = self.output_norm(fused)
        
        result = {
            'fused_features': fused,
            'vision_features': vision,
            'text_features': text,
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result


class GatedFusion(nn.Module):
    """
    Simple gated fusion for combining two modalities.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Gated fusion of two feature sets.
        
        Args:
            features1: First features [B, L, H]
            features2: Second features [B, L, H]
            
        Returns:
            Fused features [B, L, H]
        """
        combined = torch.cat([features1, features2], dim=-1)
        gate = self.gate(combined)
        return gate * features1 + (1 - gate) * features2


if __name__ == "__main__":
    # Test fusion module
    print("Testing Multimodal Fusion...")
    
    batch_size = 2
    vision_seq_len = 196  # 14x14 patches
    text_seq_len = 64
    knowledge_seq_len = 32
    
    # Create dummy inputs
    vision_features = torch.randn(batch_size, vision_seq_len, 1024)
    text_features = torch.randn(batch_size, text_seq_len, 4096)
    knowledge_features = torch.randn(batch_size, knowledge_seq_len, 768)
    
    # Create fusion module
    fusion = MultimodalFusion(
        vision_dim=1024,
        text_dim=4096,
        knowledge_dim=768,
        hidden_dim=1024,
        num_heads=16,
        num_layers=2,
        use_knowledge_gating=True
    )
    
    # Forward pass
    outputs = fusion(
        vision_features=vision_features,
        text_features=text_features,
        knowledge_features=knowledge_features,
        return_attention=True
    )
    
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Attention weights keys: {list(outputs['attention_weights'].keys())}")
    print("Multimodal Fusion test passed!")
