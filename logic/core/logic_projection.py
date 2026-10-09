from __future__ import annotations

import torch
from torch import nn


class LogicProjection(nn.Module):
    """Linear adapter from backbone hidden space to logic feature space.

    This module converts per-layer transformer activations ``[B, S, H]`` into
    logic-space activations ``[B, S, L]`` used by ``LogicStream``.
    """

    def __init__(
        self,
        hidden_dim: int,
        logic_dim: int,
        *,
        skip_learned_mapping_when_dims_match: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.logic_dim = logic_dim
        self.use_identity = bool(skip_learned_mapping_when_dims_match and hidden_dim == logic_dim)
        self.proj = nn.Identity() if self.use_identity else nn.Linear(hidden_dim, logic_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logic dimension with runtime alignment.

        Args:
            hidden_states: Tensor shaped ``[B, S, H]``.

        Returns:
            Tensor shaped ``[B, S, L]``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states with rank 3 [B,S,H], got shape={tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected last dim H={self.hidden_dim}, got H={hidden_states.size(-1)}"
            )
        if not self.use_identity:
            weight = self.proj.weight
            if hidden_states.device != weight.device or hidden_states.dtype != weight.dtype:
                hidden_states = hidden_states.to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
        return self.proj(hidden_states)
