from __future__ import annotations

import torch
from torch import nn


class FusionMLP(nn.Module):
    """Fuse backbone and logic features through a residual correction.

    Given token-level backbone features ``h_llm`` and broadcast logic features
    ``h_logic``, this module predicts a correction term with an MLP and applies
    it as:

        h_fused = h_llm + alpha * correction

    where ``alpha`` is a learnable scalar controlling logic contribution strength.
    """

    def __init__(
        self,
        hidden_dim: int,
        logic_dim: int,
        fusion_hidden_dim: int = 512,
        alpha_init: float = 0.01,
        learn_fusion_alpha: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.logic_dim = logic_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + logic_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, hidden_dim),
        )
        alpha_value = torch.tensor(alpha_init, dtype=torch.float32)
        if learn_fusion_alpha:
            self.alpha = nn.Parameter(alpha_value)
        else:
            self.register_buffer("alpha", alpha_value)

    def forward(
        self,
        llm_hidden: torch.Tensor,
        logic_hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply residual fusion to token features.

        Args:
            llm_hidden: Backbone token states ``[B, S, H]``.
            logic_hidden: Logic token states ``[B, S, L]``.
            token_mask: Optional token-validity mask ``[B, S]``.

        Returns:
            Fused token states ``[B, S, H]``.
        """
        if llm_hidden.ndim != 3 or logic_hidden.ndim != 3:
            raise ValueError(
                "Expected llm_hidden and logic_hidden as rank-3 tensors [B,S,*]"
            )
        if llm_hidden.shape[:2] != logic_hidden.shape[:2]:
            raise ValueError("Batch/sequence dims must match between llm_hidden and logic_hidden")
        if llm_hidden.size(-1) != self.hidden_dim or logic_hidden.size(-1) != self.logic_dim:
            raise ValueError("Hidden dimensions do not match module configuration")
        combined = torch.cat([llm_hidden, logic_hidden], dim=-1)
        first_linear = self.mlp[0]
        if combined.device != first_linear.weight.device or combined.dtype != first_linear.weight.dtype:
            combined = combined.to(
                device=first_linear.weight.device,
                dtype=first_linear.weight.dtype,
            )
        correction = self.mlp(combined)
        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape != llm_hidden.shape[:2]:
                raise ValueError("token_mask must be shaped [B,S]")
            correction = correction * token_mask.unsqueeze(-1).to(device=correction.device, dtype=correction.dtype)
        alpha = self.alpha.to(device=correction.device, dtype=correction.dtype)
        llm_hidden = llm_hidden.to(device=correction.device, dtype=correction.dtype)
        return llm_hidden + alpha * correction


class LinearFusionBridge(nn.Module):
    """Canonical phase-1 fusion: linear logic bridge plus residual merge.

    Fuses token states as:
        fused = llm_hidden + alpha * W(logic_hidden)
    where W maps L->H and alpha is scalar.
    """

    def __init__(
        self,
        hidden_dim: int,
        logic_dim: int,
        alpha_init: float = 0.01,
        learn_fusion_alpha: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.logic_dim = int(logic_dim)
        self.bridge = nn.Linear(self.logic_dim, self.hidden_dim)
        alpha_value = torch.tensor(alpha_init, dtype=torch.float32)
        if learn_fusion_alpha:
            self.alpha = nn.Parameter(alpha_value)
        else:
            self.register_buffer("alpha", alpha_value)

    def forward(
        self,
        llm_hidden: torch.Tensor,
        logic_hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if llm_hidden.ndim != 3 or logic_hidden.ndim != 3:
            raise ValueError("Expected llm_hidden and logic_hidden as rank-3 tensors [B,S,*]")
        if llm_hidden.shape[:2] != logic_hidden.shape[:2]:
            raise ValueError("Batch/sequence dims must match between llm_hidden and logic_hidden")
        if llm_hidden.size(-1) != self.hidden_dim or logic_hidden.size(-1) != self.logic_dim:
            raise ValueError("Hidden dimensions do not match module configuration")

        if logic_hidden.device != self.bridge.weight.device or logic_hidden.dtype != self.bridge.weight.dtype:
            logic_hidden = logic_hidden.to(device=self.bridge.weight.device, dtype=self.bridge.weight.dtype)

        projected = self.bridge(logic_hidden)
        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape != llm_hidden.shape[:2]:
                raise ValueError("token_mask must be shaped [B,S]")
            projected = projected * token_mask.unsqueeze(-1).to(device=projected.device, dtype=projected.dtype)

        alpha = self.alpha.to(device=projected.device, dtype=projected.dtype)
        llm_hidden = llm_hidden.to(device=projected.device, dtype=projected.dtype)
        return llm_hidden + alpha * projected
