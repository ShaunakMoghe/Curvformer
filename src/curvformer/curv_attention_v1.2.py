# src/curvformer/curv_attention.py  (fast-v1)
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class CurvSelfAttention(nn.Module):
    """
    Dynamic diagonal(-ish) metric attention (faster variant):
      - Groupwise scales (h * groups) instead of h * D
      - Q-only scaling by default (halve elementwise work)
      - Narrow scale range for stability and fused SDPA friendliness
    """
    def __init__(
        self,
        config,
        scale_min: float = 0.95,
        scale_max: float = 1.05,
        groups: int = 16,           # must divide head_dim
        apply_to: str = "q",        # "q", "k", or "both"
        tau: float = 16.0,          # accepted but unused (compat with patcher)
        **kwargs,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        assert self.attention_head_size % groups == 0, "groups must divide head_dim"
        self.groups = groups
        self.group_dim = self.attention_head_size // groups
        self.apply_to = apply_to

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Standard BERT projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Smaller projection: hidden -> (h * groups)
        self.scale_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.groups)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

        self.scale_min = scale_min
        self.scale_max = scale_max
        self.dropout = nn.Dropout(getattr(config, "attention_probs_dropout_prob", 0.0))

        # Encourage fastest SDPA kernels on Ampere/Lovelace
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,S,H*D) -> (B,h,S,D)
        B, S, _ = x.shape
        x = x.view(B, S, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3).contiguous()

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

        # Q,K,V (B,S,H*D)
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Heads (B,h,S,D)
        Q = self._shape_heads(q)
        K = self._shape_heads(k)
        V = self._shape_heads(v)

        B, h, S, D = Q.shape
        g, gd = self.groups, self.group_dim

        # Groupwise scales: (B,S,h*g) -> (B,h,S,g) -> broadcast to (B,h,S,D)
        s_logits = self.scale_proj(hidden_states)                    # (B,S,h*g)
        s = torch.sigmoid(s_logits) * (self.scale_max - self.scale_min) + self.scale_min
        s = s.view(B, S, h, g).permute(0, 2, 1, 3).contiguous()      # (B,h,S,g)
        s = s.unsqueeze(-1).expand(B, h, S, g, gd).reshape(B, h, S, D).contiguous()

        # Apply metric scaling (Q-only by default)
        if self.apply_to in ("q", "both"):
            Q = (Q * s).contiguous()
        if self.apply_to in ("k", "both"):
            K = (K * s).contiguous()

        # SDPA
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, V)                        # (B,h,S,D)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, S, h * D)
        return (context, attn_probs) if output_attentions else (context,)
