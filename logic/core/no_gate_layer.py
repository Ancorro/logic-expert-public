from __future__ import annotations

import torch
from torch import nn


class NoGateLayer(nn.Module):
    """Control-path layer that updates logic state without routing or gates.

    This layer is used for no-gate attribution experiments where we want a
    trainable parallel pathway but remove explicit gate/routing computations.
    It consumes token-level logic features and applies per-token updates to a
    token-level ``logic_state`` contract.
    """

    def __init__(
        self,
        logic_dim: int,
        update_type: str = "mlp",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.logic_dim = logic_dim
        if update_type not in {"linear", "mlp"}:
            raise ValueError(
                f"update_type must be one of {{'linear', 'mlp'}}, got '{update_type}'"
            )
        self.update_type = update_type

        if update_type == "linear":
            self.token_update = nn.Linear(logic_dim, logic_dim)
        else:
            hid = int(hidden_dim) if hidden_dim is not None else max(1, logic_dim)
            self.token_update = nn.Sequential(
                nn.Linear(logic_dim, hid),
                nn.GELU(),
                nn.Linear(hid, hid),
                nn.GELU(),
                nn.Linear(hid, logic_dim),
            )

        self.norm = nn.LayerNorm(logic_dim)

    def _token_update_weight(self) -> torch.Tensor:
        """Return a representative weight tensor used for dtype/device alignment."""
        if isinstance(self.token_update, nn.Linear):
            return self.token_update.weight
        # Sequential MLP; first layer is guaranteed Linear.
        return self.token_update[0].weight

    def forward(
        self,
        token_logic: torch.Tensor,
        logic_state: torch.Tensor,
        inference: bool = False,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply no-gate state update and return compatibility routing stub.

        Args:
            token_logic: Projected token features ``[B, S, L]``.
            logic_state: Running logic state ``[B, S, L]``.
            inference: Ignored; kept only for interface parity with ``LogicLayer``.

        Returns:
            ``(new_logic_state, routing_stub)`` where ``routing_stub`` is an
            all-ones tensor shaped ``[B, S, 1]`` to keep downstream logging code
            compatible with routed layers.
        """
        del inference  # kept for interface compatibility with LogicLayer
        if token_logic.ndim != 3:
            raise ValueError(
                f"Expected token_logic [B,S,L], got shape={tuple(token_logic.shape)}"
            )
        if logic_state.ndim != 3:
            raise ValueError(
                f"Expected logic_state [B,S,L], got shape={tuple(logic_state.shape)}"
            )
        if token_logic.shape != logic_state.shape:
            raise ValueError("token_logic and logic_state must have matching [B,S,L] shape")
        if token_logic.size(-1) != self.logic_dim:
            raise ValueError(f"Expected logic dim L={self.logic_dim}")
        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape != token_logic.shape[:2]:
                raise ValueError("token_mask must be shaped [B,S] and match token dims")
            mask = token_mask.to(device=token_logic.device, dtype=torch.bool)
        else:
            mask = None

        ref = self._token_update_weight()
        if token_logic.device != ref.device or token_logic.dtype != ref.dtype:
            token_logic = token_logic.to(device=ref.device, dtype=ref.dtype)

        token_updates = self.token_update(token_logic)  # [B,S,L]
        if mask is not None:
            token_updates = token_updates * mask.unsqueeze(-1).to(dtype=token_updates.dtype)
        new_logic_state = self.norm(logic_state + token_updates)
        if mask is not None:
            new_logic_state = torch.where(mask.unsqueeze(-1), new_logic_state, logic_state)

        # Placeholder routing for compatibility with downstream logging/metrics.
        routing_stub = token_logic.new_ones((token_logic.size(0), token_logic.size(1), 1))
        return new_logic_state, routing_stub
