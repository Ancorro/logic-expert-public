from __future__ import annotations

import torch
from torch import nn

from .logic_layer import LogicLayer
from .no_gate_layer import NoGateLayer


def _recommended_no_gate_hidden_dim(
    logic_dim: int,
    num_gates: int,
    gate_capacity: int,
) -> int:
    """Estimate a no-gate MLP width that roughly matches logic-layer parameter count.

    The no-gate control path is intended to be a capacity-matched baseline against the
    routed-gate logic layer. This helper computes a practical hidden width for the no-gate
    MLP so each no-gate layer has a similar parameter budget to a logic layer with
    ``logic_dim`` and ``num_gates``.

    Returns:
        An integer hidden dimension >= 1 for the no-gate update MLP.
    """
    # With operator-specific banks, routed gate channels are 3G.
    # In operator='all', gate composition emits:
    #   AND: G, OR: G, NOT: G*capacity -> total G*(capacity+2)
    # Logic-layer params (approx): router + gate scalar + update + norm
    # ~= (L*(3G)+3G) + (L+1) + (G*(capacity+2)*L + L) + (2L)
    routed_gates = 3 * num_gates
    update_in = num_gates * (gate_capacity + 2)
    target = (logic_dim * routed_gates + routed_gates) + (logic_dim + 1) + (update_in * logic_dim + logic_dim) + (2 * logic_dim)
    denom = (2 * logic_dim) + 1
    return max(1, target // denom)


class LogicStream(nn.Module):
    """Run a layer-aligned logic pathway over projected backbone hidden states.

    Overview:
        - Expects one projected tensor per backbone layer: ``projected_states[i]`` has
          shape ``[B, S, L]``.
        - Maintains a token-level logic state ``logic_state`` with shape ``[B, S, L]``.
        - Iterates through layers in order, where each layer consumes the current token
          features and the previous logic state, then returns an updated logic state.
        - Collects routing tensors from each layer for diagnostics/visualization.

    Important behavior:
        - ``logic_state`` is initialized to zeros at the start of every forward call.
          There is no persistent cross-batch state.
        - The stream can operate in two modes:
          1) standard logic layers with routing/gates (``LogicLayer``), or
          2) no-gate control layers (``NoGateLayer``) used for attribution baselines.

    Args:
        num_layers: Number of layer-aligned logic blocks to apply.
        logic_dim: Width ``L`` of the logic feature/state space.
        num_gates: Number of routed gates ``G`` per logic layer.
        routing_top_k: Inference-time top-k sparsification in routing (standard mode).
            Set to ``None`` to disable inference top-k pruning.
        routing_train_mode: Training routing mode (for example, dense or cutoff).
        routing_train_cutoff: Cutoff threshold used by cutoff training mode.
        routing_temperature: Softmax temperature for routing logits.
        gate_mode: Gate behavior in standard mode (soft or ste_binary).
        gate_operator: Gate composition mode (all/and/or/not).
        gate_capacity: Rolling gate window size used by gate composition.
        pre_routing_mlp: If True, apply a logic_dim->logic_dim MLP before routing.
        use_no_gate_stream: If True, build ``NoGateLayer`` stack instead of ``LogicLayer``.
        no_gate_update_type: Update block type for no-gate layers.
        no_gate_update_hidden_dim: Optional hidden width for no-gate MLP updates.
        no_gate_match_logic_params: If True, auto-select no-gate width to approximate
            logic-layer parameter count.

    Forward Args:
        projected_states: List of ``num_layers`` tensors, each shaped ``[B, S, L]``.
        token_mask: Optional validity mask ``[B, S]`` where True means real token.
        inference: Whether to run layers in inference mode.

    Returns:
        A tuple ``(logic_state, routing_history)`` where:
        - ``logic_state`` is the final token-level state ``[B, S, L]``.
        - ``routing_history`` is a list of per-layer routing tensors.
    """

    def __init__(
        self,
        num_layers: int,
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
        use_no_gate_stream: bool = False,
        no_gate_update_type: str = "mlp",
        no_gate_update_hidden_dim: int | None = None,
        no_gate_match_logic_params: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.logic_dim = logic_dim
        self.use_no_gate_stream = bool(use_no_gate_stream)

        if self.use_no_gate_stream:
            hidden_dim = no_gate_update_hidden_dim
            if hidden_dim is None or no_gate_match_logic_params:
                hidden_dim = _recommended_no_gate_hidden_dim(
                    logic_dim,
                    num_gates,
                    gate_capacity,
                )
            self.no_gate_update_hidden_dim = int(hidden_dim)
            self.layers = nn.ModuleList(
                [
                    NoGateLayer(
                        logic_dim=logic_dim,
                        update_type=no_gate_update_type,
                        hidden_dim=self.no_gate_update_hidden_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.no_gate_update_hidden_dim = None
            self.layers = nn.ModuleList(
                [
                    LogicLayer(
                        logic_dim,
                        num_gates,
                        routing_top_k=routing_top_k,
                        routing_train_mode=routing_train_mode,
                        routing_train_cutoff=routing_train_cutoff,
                        routing_temperature=routing_temperature,
                        gate_mode=gate_mode,
                        gate_operator=gate_operator,
                        gate_capacity=gate_capacity,
                        gate_window_axis=gate_window_axis,
                        pre_routing_mlp=pre_routing_mlp,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(
        self,
        projected_states: list[torch.Tensor],
        inference: bool = False,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(projected_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} projected states, got {len(projected_states)}"
            )
        first = projected_states[0]
        if first.ndim != 3:
            raise ValueError(
                f"Projected state at index 0 must be [B,S,L], got {tuple(first.shape)}"
            )
        batch_size, seq_len, logic_dim = first.shape
        if logic_dim != self.logic_dim:
            raise ValueError(f"Expected logic dim L={self.logic_dim}, got {logic_dim}")

        if token_mask is not None:
            if token_mask.ndim != 2 or token_mask.shape != (batch_size, seq_len):
                raise ValueError("token_mask must be shaped [B,S] and match projected states")
            mask = token_mask.to(device=first.device, dtype=torch.bool)
        else:
            mask = None

        logic_state = first.new_zeros((batch_size, seq_len, self.logic_dim))

        routing_history: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            token_logic = projected_states[layer_idx]
            if token_logic.ndim != 3:
                raise ValueError(
                    f"Projected state at index {layer_idx} must be [B,S,L], got {tuple(token_logic.shape)}"
                )
            logic_state, routing_weights = layer(
                token_logic,
                logic_state,
                inference=inference,
                token_mask=mask,
            )
            routing_history.append(routing_weights)

        return logic_state, routing_history
