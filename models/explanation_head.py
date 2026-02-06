"""
Explanation Head Module
=======================
Generates textual rationales and explanations for VQA predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class ExplanationHead(nn.Module):
    """
    Explanation head for generating textual rationales.
    
    Produces coherent explanations for VQA predictions based on
    fused multimodal representations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        vocab_size: int = 32000,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        """
        Initialize explanation head.
        
        Args:
            hidden_dim: Hidden dimension
            vocab_size: Vocabulary size
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            max_length: Maximum explanation length
            dropout: Dropout probability
            tie_weights: Tie input/output embeddings
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights
        if tie_weights:
            self.output_proj.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _generate_causal_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        return mask
    
    def forward(
        self,
        context: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            context: Fused multimodal features [B, L, H]
            target_ids: Target token IDs for teacher forcing [B, T]
            target_mask: Target attention mask [B, T]
            memory_mask: Mask for context [B, L]
            
        Returns:
            Dictionary with logits and hidden states
        """
        if target_ids is None:
            # Generate mode - start with BOS token (typically 1)
            return self.generate(context, memory_mask=memory_mask)
        
        batch_size, seq_len = target_ids.shape
        device = target_ids.device
        
        # Embed tokens
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_emb = self.token_embedding(target_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Decode
        hidden_states = self.decoder(
            x,
            context,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~target_mask.bool() if target_mask is not None else None,
            memory_key_padding_mask=~memory_mask.bool() if memory_mask is not None else None,
        )
        
        # Output projection
        hidden_states = self.output_norm(hidden_states)
        logits = self.output_proj(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def generate(
        self,
        context: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate explanation autoregressively.
        
        Args:
            context: Fused features [B, L, H]
            memory_mask: Context mask [B, L]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Use sampling (vs greedy)
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            
        Returns:
            Generated token IDs and scores
        """
        max_length = max_length or self.max_length
        batch_size = context.shape[0]
        device = context.device
        
        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scores = []
        
        for step in range(max_length - 1):
            if done.all():
                break
            
            # Get logits for current sequence
            outputs = self.forward(
                context=context,
                target_ids=generated,
                target_mask=torch.ones_like(generated),
                memory_mask=memory_mask
            )
            
            # Get logits for last position
            next_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if do_sample:
                next_logits = self._top_k_top_p_filter(next_logits, top_k, top_p)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Track scores
            scores.append(F.log_softmax(outputs['logits'][:, -1, :], dim=-1))
            
            # Update done status
            done = done | (next_token.squeeze(-1) == eos_token_id)
            
            # Append next token
            next_token = next_token.masked_fill(done.unsqueeze(-1), pad_token_id)
            generated = torch.cat([generated, next_token], dim=1)
        
        return {
            'generated_ids': generated,
            'scores': torch.stack(scores, dim=1) if scores else None
        }
    
    def _top_k_top_p_filter(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Apply top-k and top-p (nucleus) filtering."""
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits


class RationaleGenerator(nn.Module):
    """
    Generates structured medical rationales.
    
    Produces explanations following a specific medical reasoning pattern:
    1. Observation: What is observed in the image
    2. Analysis: Medical interpretation
    3. Conclusion: Answer rationale
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 2048,
        num_reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize rationale generator.
        
        Args:
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate dimension
            num_reasoning_steps: Number of reasoning steps
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps
        
        # Step embeddings
        self.step_embedding = nn.Embedding(num_reasoning_steps, hidden_dim)
        
        # Reasoning layers (one per step)
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, intermediate_dim),
                nn.LayerNorm(intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_reasoning_steps)
        ])
        
        # Step attention (attend to relevant parts of context for each step)
        self.step_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_reasoning_steps)
        ])
        
        # Output projections for each step
        self.step_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_reasoning_steps)
        ])
    
    def forward(
        self,
        fused_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate structured rationale representations.
        
        Args:
            fused_features: Fused multimodal features [B, L, H]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Step-wise rationale representations
        """
        batch_size = fused_features.shape[0]
        device = fused_features.device
        
        # Get pooled context
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            context_sum = (fused_features * mask_expanded).sum(dim=1)
            context_count = mask_expanded.sum(dim=1).clamp(min=1)
            context = context_sum / context_count
        else:
            context = fused_features.mean(dim=1)
        
        step_outputs = []
        current_state = context
        
        for step_idx in range(self.num_reasoning_steps):
            # Get step embedding
            step_emb = self.step_embedding(
                torch.tensor([step_idx], device=device)
            ).expand(batch_size, -1)
            
            # Attend to fused features
            step_query = (current_state + step_emb).unsqueeze(1)
            attended, attn_weights = self.step_attention[step_idx](
                step_query,
                fused_features,
                fused_features,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            attended = attended.squeeze(1)
            
            # Reasoning step
            combined = torch.cat([current_state, attended], dim=-1)
            step_state = self.reasoning_layers[step_idx](combined)
            
            # Residual connection
            current_state = current_state + step_state
            
            # Output for this step
            step_output = self.step_outputs[step_idx](current_state)
            step_outputs.append(step_output)
        
        return {
            'step_representations': torch.stack(step_outputs, dim=1),  # [B, num_steps, H]
            'final_state': current_state,  # [B, H]
            'observation': step_outputs[0],  # [B, H]
            'analysis': step_outputs[1] if len(step_outputs) > 1 else None,  # [B, H]
            'conclusion': step_outputs[-1],  # [B, H]
        }


class ExplanationModule(nn.Module):
    """
    Complete explanation module combining rationale generation and text output.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        vocab_size: int = 32000,
        max_length: int = 128,
        use_structured_rationale: bool = True,
    ):
        """
        Initialize explanation module.
        
        Args:
            hidden_dim: Hidden dimension
            vocab_size: Vocabulary size
            max_length: Maximum explanation length
            use_structured_rationale: Use structured rationale generator
        """
        super().__init__()
        
        self.use_structured_rationale = use_structured_rationale
        
        # Rationale generator
        if use_structured_rationale:
            self.rationale_generator = RationaleGenerator(hidden_dim=hidden_dim)
        
        # Text explanation head
        self.explanation_head = ExplanationHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_length=max_length
        )
        
        # Combine rationale with context
        if use_structured_rationale:
            self.rationale_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
    
    def forward(
        self,
        fused_features: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate explanations.
        
        Args:
            fused_features: Fused multimodal features [B, L, H]
            target_ids: Target explanation tokens [B, T]
            target_mask: Target mask [B, T]
            attention_mask: Context attention mask [B, L]
            
        Returns:
            Explanation outputs
        """
        results = {}
        
        # Generate structured rationale
        if self.use_structured_rationale:
            rationale_outputs = self.rationale_generator(
                fused_features=fused_features,
                attention_mask=attention_mask
            )
            results['rationale'] = rationale_outputs
            
            # Fuse rationale with context
            rationale_expanded = rationale_outputs['final_state'].unsqueeze(1)
            rationale_expanded = rationale_expanded.expand(-1, fused_features.shape[1], -1)
            context = self.rationale_fusion(
                torch.cat([fused_features, rationale_expanded], dim=-1)
            )
        else:
            context = fused_features
        
        # Generate text explanation
        explanation_outputs = self.explanation_head(
            context=context,
            target_ids=target_ids,
            target_mask=target_mask,
            memory_mask=attention_mask
        )
        results.update(explanation_outputs)
        
        return results
    
    def generate(
        self,
        fused_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate explanation text."""
        # Get rationale
        if self.use_structured_rationale:
            rationale_outputs = self.rationale_generator(
                fused_features=fused_features,
                attention_mask=attention_mask
            )
            rationale_expanded = rationale_outputs['final_state'].unsqueeze(1)
            rationale_expanded = rationale_expanded.expand(-1, fused_features.shape[1], -1)
            context = self.rationale_fusion(
                torch.cat([fused_features, rationale_expanded], dim=-1)
            )
        else:
            context = fused_features
        
        # Generate
        return self.explanation_head.generate(
            context=context,
            memory_mask=attention_mask,
            **generate_kwargs
        )


if __name__ == "__main__":
    # Test explanation head
    print("Testing Explanation Head...")
    
    batch_size = 2
    seq_len = 64
    hidden_dim = 1024
    vocab_size = 32000
    
    # Create dummy inputs
    context = torch.randn(batch_size, seq_len, hidden_dim)
    target_ids = torch.randint(0, vocab_size, (batch_size, 32))
    target_mask = torch.ones(batch_size, 32)
    
    # Create explanation module
    module = ExplanationModule(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        use_structured_rationale=True
    )
    
    # Forward pass
    outputs = module(
        fused_features=context,
        target_ids=target_ids,
        target_mask=target_mask
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Rationale steps shape: {outputs['rationale']['step_representations'].shape}")
    
    # Test generation
    gen_outputs = module.generate(context, max_length=20)
    print(f"Generated shape: {gen_outputs['generated_ids'].shape}")
    print("Explanation Head test passed!")
