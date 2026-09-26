"""Naive layer-wise model parallelism for a HF decoder backbone (opt-in).

Splits the backbone's transformer blocks evenly across ``devices`` in one process, so the
computation is identical to single-GPU training (no sharding or distributed reductions).
It is used only when an 8B full fine-tune does not fit on one GPU. The logic stream, fusion
and task head stay on ``devices[0]``; callers move hidden states there.
"""

from __future__ import annotations

from torch import nn


def dispatch_backbone(backbone: nn.Module, devices: list[str]) -> str:
    """Place embed/first layers on devices[0] ... last layers/norm on devices[-1].

    Returns the device that the logic modules and task head should live on.
    """
    from accelerate import dispatch_model

    owner = getattr(backbone, "model", backbone)  # CausalLM wraps the decoder in .model
    prefix = "model." if owner is not backbone else ""
    n_layers = len(owner.layers)
    per_dev = -(-n_layers // len(devices))
    device_map = {f"{prefix}embed_tokens": devices[0], f"{prefix}rotary_emb": devices[0]}
    for i in range(n_layers):
        device_map[f"{prefix}layers.{i}"] = devices[i // per_dev]
    device_map[f"{prefix}norm"] = devices[-1]
    dispatch_model(backbone, device_map=device_map)
    return devices[0]
