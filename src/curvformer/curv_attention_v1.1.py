# src/curvformer/curv_attention.py
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class CurvSelfAttention(nn.Module):
    """
    One-path dynamic diagonal(-ish) metric attention, optimized for speed:
      - Groupwise scales to cut projection cost
      - Optional single-sided scaling (Q or K) to halve elementwise work
    """

    def __init__(self, config,
        scale_min: float = 0.90,
        scale_max: float = 1.10,
        groups: int = 8,
        apply_to: str = "q",
        tau: float = 16.0,      # <--- absorb tau from patching.py
        **kwargs
    ):              # <--- future-proof against extra kwargs
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        assert self.attention_head_size % groups == 0, "groups must divide head_dim"
        self.groups = groups
        self.group_dim = self.attention_head_size // groups
        self.apply_to = apply_to

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Standard BERT projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # FAST: project hidden_states -> (h * groups) scales (much smaller than h * D)
        self.scale_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.groups)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.0))

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,S,H*D) -> (B,h,S,D)
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Q, K, V
        q = self.query(hidden_states)        # (B,S,H*D)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        Q = self._shape_heads(q)             # (B,h,S,D)
        K = self._shape_heads(k)
        V = self._shape_heads(v)

        B, h, S, D = Q.shape
        g = self.groups
        gd = self.group_dim

        # Groupwise scales: (B,S,h*g) -> (B,h,S,g) -> expand to (B,h,S,D)
        s_logits = self.scale_proj(hidden_states)                           # (B,S,h*g)
        s = torch.sigmoid(s_logits) * (self.scale_max - self.scale_min) + self.scale_min
        s = s.view(B, S, h, g).permute(0, 2, 1, 3).contiguous()             # (B,h,S,g)
        s = s.unsqueeze(-1).expand(B, h, S, g, gd).reshape(B, h, S, D)      # (B,h,S,D)

        # Apply metric scaling (single-sided by default for speed)
        if self.apply_to in ("q", "both"):
            Q = Q * s
        if self.apply_to in ("k", "both"):
            K = K * s

        # SDPA
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)     # (B,h,S,S)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, V)                               # (B,h,S,D)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, S, h * D)
        return (context, attn_probs) if output_attentions else (context,)
