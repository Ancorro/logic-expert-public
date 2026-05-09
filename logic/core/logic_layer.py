from __future__ import annotations

import torch
from torch import nn

from .logic_gates import compose_all_gate_outputs, compose_gate_outputs
from .routing import RoutingModule


class LogicLayer(nn.Module):
    """
    One routed logic update block operating on a layer-aligned token tensor.

    Pipeline:
        token_logic [B,S,L]
                    -> routing weights [B,S,3G]
                                        -> gate values [B,S,3G]
                    -> composed gate outputs [B,S,*]
                    -> logic delta [B,S,L]
                    -> normalized updated state [B,S,L]

    The layer updates only ``logic_state``; it does not mutate backbone hidden
    states directly.
    """

    def __init__(
        self,
        logic_dim: int,
        num_gates: int,
        routing_top_k: int | None = 2,
        routing_train_mode: str = "dense",
        routing_train_cutoff: float = 0.0,
        routing_temperature: float = 1.0,
        gate_mode: str = "soft",
        gate_operator: str = "all",
        gate_capacity: int = 2,
        gate_window_axis: str = "inter_token",
        pre_routing_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.logic_dim = logic_dim
        if num_gates < 1:
            raise ValueError(f"num_gates must be >= 1, got {num_gates}")
        self.num_gates = int(num_gates)
        self.total_num_gates = int(3 * self.num_gates)
        if gate_mode not in {"soft", "ste_binary"}:
            raise ValueError(
                f"gate_mode must be one of {{'soft', 'ste_binary'}}, got '{gate_mode}'"
            )
        if gate_operator not in {"all", "and", "or", "not"}:
            raise ValueError(
                f"gate_operator must be one of {{'all', 'and', 'or', 'not'}}, got '{gate_operator}'"
            )
        if gate_capacity < 1:
            raise ValueError(f"gate_capacity must be >= 1, got {gate_capacity}")
        if gate_window_axis not in {"inter_token", "intra_token"}:
            raise ValueError(
                f"gate_window_axis must be one of {{'inter_token', 'intra_token'}}, got '{gate_window_axis}'"
            )
        self.gate_mode = gate_mode
        self.gate_operator = gate_operator
        self.gate_capacity = int(gate_capacity)
        self.gate_window_axis = gate_window_axis
        self.pre_routing_mlp_enabled = bool(pre_routing_mlp)
        if self.pre_routing_mlp_enabled:
            # Optional adapter before routing: [B,S,L] -> [B,S,L]
            self.pre_routing_mlp = nn.Sequential(
                nn.Linear(logic_dim, logic_dim),
                nn.GELU(),
                nn.Linear(logic_dim, logic_dim),
                nn.GELU(),
                nn.Linear(logic_dim, logic_dim),
            )
        else:
            self.pre_routing_mlp = nn.Identity()
        self.routing = RoutingModule(
            logic_dim,
            self.total_num_gates,
            top_k=routing_top_k,
            train_mode=routing_train_mode,
            train_cutoff=routing_train_cutoff,
            temperature=routing_temperature,
        )
        self.gate_scalar = nn.Linear(logic_dim, self.total_num_gates)
        if self.gate_operator == "all":
            logic_update_in_dim = self.num_gates * (self.gate_capacity + 2)
        elif self.gate_operator == "not":
            logic_update_in_dim = self.num_gates * self.gate_capacity
        else:
            logic_update_in_dim = self.num_gates
        self.logic_update = nn.Linear(logic_update_in_dim, logic_dim)
        if hasattr(nn, "RMSNorm"):
            self.pre_update_norm = nn.RMSNorm(logic_dim)
            self.post_residual_norm = nn.RMSNorm(logic_dim)
        else:
            self.pre_update_norm = nn.LayerNorm(logic_dim)
            self.post_residual_norm = nn.LayerNorm(logic_dim)

    @staticmethod
    def _ste_binarize(gates: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Straight-through binarization for gates.

        Forward pass uses a hard threshold, while backward gradients flow as if
        the soft gate values were used (identity estimator).
        """
        hard = (gates >= threshold).to(dtype=gates.dtype)
        return hard.detach() - gates.detach() + gates

    def forward(
        self,
        token_logic: torch.Tensor,
        logic_state: torch.Tensor,
        inference: bool = False,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one logic-state update from token features.

        Args:
            token_logic: Token-level logic features ``[B, S, L]``.
            logic_state: Current token-level logic state ``[B, S, L]``.
            inference: Whether routing should run in inference (top-k sparse) mode.

        Returns:
            ``(new_logic_state, routing_weights)`` where routing weights have
            shape ``[B, S, 3G]``.
        """
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

        update_in = self.pre_update_norm(token_logic)
        if mask is not None:
            update_in = update_in * mask.unsqueeze(-1).to(dtype=update_in.dtype)

        routing_input = self.pre_routing_mlp(update_in)
        routing_weights = self.routing(routing_input, inference=inference, token_mask=mask)  # [B,S,3G]
        gate_values = torch.sigmoid(self.gate_scalar(routing_input))  # [B,S,3G]
        gate_values = gate_values * routing_weights
        if self.gate_mode == "ste_binary":
            gate_values = self._ste_binarize(gate_values)
        and_values, or_values, not_values = torch.split(gate_values, self.num_gates, dim=-1)
        if self.gate_operator == "all":
            gate_outputs = compose_all_gate_outputs(
                and_values,
                or_values,
                not_values,
                capacity=self.gate_capacity,
                window_axis=self.gate_window_axis,
                token_mask=mask,
            )  # [B,G*(capacity+2)]
        elif self.gate_operator == "and":
            gate_outputs = compose_gate_outputs(
                and_values,
                operator="and",
                capacity=self.gate_capacity,
                window_axis=self.gate_window_axis,
                token_mask=mask,
            )  # [B,G]
        elif self.gate_operator == "or":
            gate_outputs = compose_gate_outputs(
                or_values,
                operator="or",
                capacity=self.gate_capacity,
                window_axis=self.gate_window_axis,
                token_mask=mask,
            )  # [B,G]
        else:
            gate_outputs = compose_gate_outputs(
                not_values,
                operator="not",
                capacity=self.gate_capacity,
                window_axis=self.gate_window_axis,
                token_mask=mask,
            )  # [B,G*capacity]
        gate_outputs = gate_outputs.to(
            device=self.logic_update.weight.device,
            dtype=self.logic_update.weight.dtype,
        )

        logic_delta = self.logic_update(gate_outputs)  # [B,S,L]
        if mask is not None:
            logic_delta = logic_delta * mask.unsqueeze(-1).to(dtype=logic_delta.dtype)

        new_logic_state = self.post_residual_norm(logic_state + logic_delta)
        if mask is not None:
            mask_f = mask.unsqueeze(-1)
            new_logic_state = torch.where(mask_f, new_logic_state, logic_state)
        return new_logic_state, routing_weights
