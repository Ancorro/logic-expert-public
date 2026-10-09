from __future__ import annotations

import math

import torch
from torch import nn


class LayerwiseLogicCrossAttention(nn.Module):
    """Logic-query cross-attention block with option-1 projection space.

    Q is formed from logic tokens in L-space, while K/V are projected from
    backbone hidden states H->L. Output remains in L-space after W_o.
    """

    def __init__(
        self,
        logic_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if logic_dim % num_heads != 0:
            raise ValueError("logic_dim must be divisible by num_heads")

        self.logic_dim = int(logic_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.logic_dim // self.num_heads

        self.q_proj = nn.Linear(self.logic_dim, self.logic_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.logic_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.logic_dim)
        self.o_proj = nn.Linear(self.logic_dim, self.logic_dim)
        self.dropout = nn.Dropout(float(dropout))

        if hasattr(nn, "RMSNorm"):
            self.pre_attn_norm = nn.RMSNorm(self.logic_dim)
        else:
            self.pre_attn_norm = nn.LayerNorm(self.logic_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,S,L] -> [B,heads,S,head_dim]
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,heads,S,head_dim] -> [B,S,L]
        b, _, s, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.logic_dim)

    def forward(
        self,
        logic_state: torch.Tensor,
        backbone_hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logic_state.ndim != 3 or backbone_hidden.ndim != 3:
            raise ValueError("logic_state and backbone_hidden must be rank-3 tensors")
        if logic_state.shape[:2] != backbone_hidden.shape[:2]:
            raise ValueError("Batch/sequence dimensions must match")
        if logic_state.size(-1) != self.logic_dim:
            raise ValueError("logic_state last dim must match logic_dim")
        if backbone_hidden.size(-1) != self.hidden_dim:
            raise ValueError("backbone_hidden last dim must match hidden_dim")

        bsz, seq_len, _ = logic_state.shape
        mask = None
        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape != (bsz, seq_len):
                raise ValueError("token_mask must be shaped [B,S]")
            mask = token_mask.to(device=logic_state.device, dtype=torch.bool)

        q_in = self.pre_attn_norm(logic_state)
        q = self._split_heads(self.q_proj(q_in))
        k = self._split_heads(self.k_proj(backbone_hidden))
        v = self._split_heads(self.v_proj(backbone_hidden))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            key_mask = mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn_probs = torch.softmax(scores, dim=-1)
        attn_ctx = torch.matmul(attn_probs, v)

        if mask is not None:
            query_mask = mask[:, None, :, None]
            attn_ctx = attn_ctx * query_mask.to(dtype=attn_ctx.dtype)

        attended = self._merge_heads(attn_ctx)
        attended = self.o_proj(attended)
        attended = self.dropout(attended)

        # Residual in logic space, preserving previous state for padded positions.
        updated = logic_state + attended
        if mask is not None:
            updated = torch.where(mask.unsqueeze(-1), updated, logic_state)

        return updated, attn_probs
