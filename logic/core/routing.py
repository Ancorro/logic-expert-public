from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RoutingModule(nn.Module):
    """Map token-level logic features to per-token gate weights.

    Input tensor shape is ``[B, S, L]`` where ``L`` is logic feature width.
    Output tensor shape is ``[B, S, G]`` where ``G`` is number of gates.

    Behavior by mode:
    - Training (default): returns dense softmax routing, or cutoff-pruned routing if
      ``train_mode='cutoff'`` and ``train_cutoff > 0``.
        - Inference: applies top-k sparsification per token, then renormalizes.
            If ``top_k=None`` or ``top_k='all'``, inference uses dense softmax routing
            (no top-k pruning).
    """

    def __init__(
        self,
        logic_dim: int,
        num_gates: int,
        top_k: int | str | None = 2,
        train_mode: str = "dense",
        train_cutoff: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(top_k, str):
            if top_k.lower() == "all":
                top_k = None
            else:
                raise ValueError(
                    f"top_k string must be 'all' when provided, got '{top_k}'"
                )
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be >=1, got {top_k}")
        if train_mode not in {"dense", "cutoff"}:
            raise ValueError(
                f"train_mode must be one of {{'dense', 'cutoff'}}, got '{train_mode}'"
            )
        if train_cutoff < 0.0 or train_cutoff >= 1.0:
            raise ValueError(
                f"train_cutoff must be in [0,1), got {train_cutoff}"
            )
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )
        self.logic_dim = logic_dim
        self.num_gates = num_gates
        self.top_k = top_k
        self.train_mode = train_mode
        self.train_cutoff = train_cutoff
        self.temperature = temperature
        self.router = nn.Linear(logic_dim, num_gates)

    @staticmethod
    def _apply_cutoff(weights: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Apply probability cutoff with safe renormalization.

        Any gate weight below ``cutoff`` is set to zero. Remaining weights are
        renormalized per token. If all gates are removed for a token, the function
        falls back to a one-hot vector at the original argmax gate.
        """
        masked = torch.where(weights >= cutoff, weights, torch.zeros_like(weights))
        denom = masked.sum(dim=-1, keepdim=True)

        # Avoid all-zero rows by falling back to the strongest original gate.
        argmax_idx = weights.argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(weights).scatter_(-1, argmax_idx, 1.0)
        normalized = masked / denom.clamp_min(1e-8)
        return torch.where(denom > 0.0, normalized, fallback)

    def forward(
        self,
        token_logic: torch.Tensor,
        inference: bool = False,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute routing weights for each token.

        Args:
            token_logic: Logic features shaped ``[B, S, L]``.
            inference: If True, apply top-k sparse routing; otherwise training behavior.

        Returns:
            Routing tensor shaped ``[B, S, G]``.
        """
        if token_logic.ndim != 3:
            raise ValueError(
                f"Expected token_logic with rank 3 [B,S,L], got shape={tuple(token_logic.shape)}"
            )
        if token_logic.size(-1) != self.logic_dim:
            raise ValueError(
                f"Expected last dim L={self.logic_dim}, got L={token_logic.size(-1)}"
            )
        if token_mask is not None:
            if token_mask.ndim != 2:
                raise ValueError(
                    f"Expected token_mask [B,S], got shape={tuple(token_mask.shape)}"
                )
            if token_mask.shape != token_logic.shape[:2]:
                raise ValueError("token_mask shape must match token_logic batch/seq dims")

        if token_logic.device != self.router.weight.device or token_logic.dtype != self.router.weight.dtype:
            token_logic = token_logic.to(
                device=self.router.weight.device,
                dtype=self.router.weight.dtype,
            )
        mask = None
        if token_mask is not None:
            mask = token_mask.to(device=token_logic.device, dtype=torch.bool)

        scores = self.router(token_logic)  # [B,S,G]
        scores = scores / self.temperature
        weights = F.softmax(scores, dim=-1)
        if not inference:
            if self.train_mode == "cutoff" and self.train_cutoff > 0.0:
                weights = self._apply_cutoff(weights, self.train_cutoff)
            if mask is None:
                return weights
            return weights * mask.unsqueeze(-1).to(dtype=weights.dtype)

        if self.top_k is None:
            if mask is None:
                return weights
            return weights * mask.unsqueeze(-1).to(dtype=weights.dtype)

        k = min(self.top_k, self.num_gates)
        values, indices = torch.topk(weights, k=k, dim=-1)
        sparse = torch.zeros_like(weights).scatter_(-1, indices, values)
        denom = sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sparse = sparse / denom
        if mask is None:
            return sparse
        return sparse * mask.unsqueeze(-1).to(dtype=sparse.dtype)
