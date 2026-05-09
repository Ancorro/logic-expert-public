from __future__ import annotations

import torch


def _clip(x: torch.Tensor) -> torch.Tensor:
    """Bound values to [0, 1] for fuzzy/probabilistic gate math."""
    return torch.clamp(x, min=0.0, max=1.0)


def and_gate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fuzzy AND using elementwise product after clipping inputs to [0, 1]."""
    a, b = _clip(a), _clip(b)
    return a * b


def or_gate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fuzzy OR using a + b - a*b after clipping inputs to [0, 1]."""
    a, b = _clip(a), _clip(b)
    return a + b - a * b


def not_gate(a: torch.Tensor) -> torch.Tensor:
    """Fuzzy NOT computed as 1 - a after clipping to [0, 1]."""
    return torch.ones_like(a) - _clip(a)


def _shift_prev_tokens(gate_values: torch.Tensor, shift: int) -> torch.Tensor:
    """Causal token shift along S without wrap-around for tensors shaped [B,S,G]."""
    if shift == 0:
        return gate_values
    out = torch.zeros_like(gate_values)
    out[:, shift:, :] = gate_values[:, :-shift, :]
    return out


def _shift_prev_channels(gate_values: torch.Tensor, shift: int) -> torch.Tensor:
    """Causal channel shift along G without wrap-around for tensors shaped [B,S,G]."""
    if shift == 0:
        return gate_values
    out = torch.zeros_like(gate_values)
    out[:, :, shift:] = gate_values[:, :, :-shift]
    return out


def _token_window_stack(
    gate_values: torch.Tensor,
    capacity: int,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build causal token windows [B,S,G,capacity] and optional valid mask [B,S,1,capacity]."""
    windows = [_shift_prev_tokens(gate_values, shift=shift) for shift in range(capacity)]
    stacked = torch.stack(windows, dim=-1)

    if token_mask is None:
        return stacked, None

    if token_mask.ndim != 2 or token_mask.shape != gate_values.shape[:2]:
        raise ValueError(
            "token_mask must be shaped [B,S] and match gate tensor token dims"
        )
    mask = token_mask.to(device=gate_values.device, dtype=torch.bool)
    shifted_masks = [_shift_prev_tokens(mask.unsqueeze(-1).to(gate_values.dtype), shift=shift) for shift in range(capacity)]
    mask_stacked = torch.stack(shifted_masks, dim=-1).to(dtype=torch.bool)
    return stacked, mask_stacked


def _channel_window_stack(
    gate_values: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, None]:
    """Build causal channel windows [B,S,G,capacity] along gate axis (intra-token)."""
    windows = [_shift_prev_channels(gate_values, shift=shift) for shift in range(capacity)]
    stacked = torch.stack(windows, dim=-1)
    return stacked, None


def compose_gate_outputs(
    gate_values: torch.Tensor,
    operator: str,
    capacity: int = 2,
    window_axis: str = "inter_token",
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compose fixed gate transforms into a bounded gate-output summary.

    Given gate activations ``gate_values`` with shape ``[B, S, G]``, this function
    applies one logical operator over a causal window of size ``capacity``:
    - ``and``: all values in the token window must be high
    - ``or``: at least one value in the token window is high
    - ``not``: NOT over every element in each token window

    ``window_axis`` selects how windows are formed:
    - ``inter_token``: causal windows over prior tokens (current behavior)
    - ``intra_token``: causal windows over gate channels within each token

    ``capacity`` is a hyperparameter controlling how many rolled inputs each gate
    operation sees.

    ``token_mask`` (shape ``[B, S]``) marks valid tokens and prevents padding from
    contributing to window reductions.

    Returns:
        - ``and``/``or``: tensor ``[..., G]``
        - ``not``: tensor ``[..., G * capacity]``
    """
    if gate_values.ndim != 3:
        raise ValueError(
            f"Expected gate_values [B,S,G], got shape={tuple(gate_values.shape)}"
        )
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")

    gate_values = _clip(gate_values)
    if window_axis == "inter_token":
        windows, valid = _token_window_stack(gate_values, capacity, token_mask=token_mask)
    elif window_axis == "intra_token":
        windows, valid = _channel_window_stack(gate_values, capacity)
    else:
        raise ValueError("window_axis must be one of {'inter_token', 'intra_token'}")

    if valid is None:
        and_inputs = windows
        or_inputs = windows
        not_inputs = not_gate(windows)
    else:
        and_inputs = torch.where(valid, windows, torch.ones_like(windows))
        or_inputs = torch.where(valid, windows, torch.zeros_like(windows))
        not_inputs = torch.where(valid, not_gate(windows), torch.zeros_like(windows))

    and_out = torch.prod(and_inputs, dim=-1)
    or_out = 1.0 - torch.prod(1.0 - or_inputs, dim=-1)
    not_out = not_inputs.reshape(*gate_values.shape[:-1], -1)

    if operator == "and":
        return and_out
    if operator == "or":
        return or_out
    if operator == "not":
        return not_out
    raise ValueError("operator must be one of {'and', 'or', 'not'}")


def compose_all_gate_outputs(
    and_gate_values: torch.Tensor,
    or_gate_values: torch.Tensor,
    not_gate_values: torch.Tensor,
    capacity: int = 2,
    window_axis: str = "inter_token",
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose independent AND/OR/NOT gate banks and concatenate outputs.

    Each input is ``[..., G]`` where ``G`` is gates per operator.
    Returns ``[..., G * (capacity + 2)]`` in the order:
    ``[AND(G), OR(G), NOT(G*capacity)]``.
    """
    and_out = compose_gate_outputs(
        and_gate_values,
        operator="and",
        capacity=capacity,
        window_axis=window_axis,
        token_mask=token_mask,
    )
    or_out = compose_gate_outputs(
        or_gate_values,
        operator="or",
        capacity=capacity,
        window_axis=window_axis,
        token_mask=token_mask,
    )
    not_out = compose_gate_outputs(
        not_gate_values,
        operator="not",
        capacity=capacity,
        window_axis=window_axis,
        token_mask=token_mask,
    )
    return torch.cat([and_out, or_out, not_out], dim=-1)
