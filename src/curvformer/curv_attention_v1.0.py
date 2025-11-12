# src/curvformer/curv_attention.py
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class CurvSelfAttention(nn.Module):
    """
    One-path dynamic diagonal-metric self-attention.

    Idea: before the usual QK^T / sqrt(d) logits, scale Q and K elementwise by a
    per-token, per-head, per-dim positive vector s, derived from the current hidden states.
      Q' = Q ⊙ s,   K' = K ⊙ s
      S  = (Q' K'^T) / sqrt(d)

    - No Euclidean/“hyperbolic” mixing branch.
    - Keeps SDPA/FlashAttention compatibility and speed.
    """
    def __init__(self, config, tau: float = 16.0, scale_min: float = 0.5, scale_max: float = 1.5):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Standard BERT projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dynamic diagonal scale s (token-, head-, and dim-wise): (B, S, H*D) → (B, h, S, D)
        # NOTE: zero init + [scale_min, scale_max] via sigmoid makes initial s ≈ 1.0 (stable).
        self.scale_proj = nn.Linear(config.hidden_size, self.all_head_size)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.0))

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, H*D) -> (B, h, S, D)
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,  # cache_position / is_causal / etc.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Q, K, V (B, S, H*D)
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Heads: (B, h, S, D)
        Q = self._shape_heads(q)
        K = self._shape_heads(k)
        V = self._shape_heads(v)

        # Dynamic per-token, per-head, per-dim positive scale s in [scale_min, scale_max]
        # s_logits: (B, S, H*D) -> s_shaped: (B, h, S, D)
        s_logits = self.scale_proj(hidden_states)
        s = torch.sigmoid(s_logits) * (self.scale_max - self.scale_min) + self.scale_min
        s = s.view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.attention_head_size)
        s = s.permute(0, 2, 1, 3).contiguous()  # (B, h, S, D)

        # Apply the diagonal metric: Q' = Q ⊙ s, K' = K ⊙ s
        Q = Q * s
        K = K * s

        # Scaled dot-product logits
        sqrt_d = self.attention_head_size ** 0.5
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / sqrt_d  # (B, h, S, S)

        # Mask + softmax
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_probs = self.dropout(attn_probs)

        # Context
        context = torch.matmul(attn_probs, V)  # (B, h, S, D)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)  # (B, S, H*D)

        return (context, attn_probs) if output_attentions else (context,)
