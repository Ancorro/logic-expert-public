"""Naive layer-wise model parallelism for a HF decoder backbone (opt-in).

Splits the backbone's transformer blocks evenly across ``devices`` in one process, so the
computation is identical to single-GPU training (no sharding or distributed reductions).
It is used only when an 8B full fine-tune does not fit on one GPU. Embeddings and rotary
tables stay on ``devices[0]``, and each block/norm gets a forward pre-hook that moves its
inputs to its own device. The logic stream, fusion and task head stay on ``devices[0]``;
callers move hidden states there.
"""

from __future__ import annotations

import torch
from torch import nn


def _to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to(x, device) for x in obj)
    if isinstance(obj, list):
        return [_to(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def _move_inputs_hook(device):
    def _hook(_module, args, kwargs):
        return _to(args, device), _to(kwargs, device)

    return _hook


def dispatch_backbone(backbone: nn.Module, devices: list[str]) -> str:
    """Place embed/first layers on devices[0] ... last layers/norm on devices[-1].

    Returns the device that the logic modules and task head should live on.
    """
    owner = getattr(backbone, "model", backbone)  # CausalLM wraps the decoder in .model
    layers = owner.layers
    per_dev = -(-len(layers) // len(devices))
    owner.embed_tokens.to(devices[0])
    if getattr(owner, "rotary_emb", None) is not None:
        owner.rotary_emb.to(devices[0])
    for i, layer in enumerate(layers):
        dev = devices[i // per_dev]
        layer.to(dev)
        layer.register_forward_pre_hook(_move_inputs_hook(dev), with_kwargs=True)
    owner.norm.to(devices[-1])
    owner.norm.register_forward_pre_hook(_move_inputs_hook(devices[-1]), with_kwargs=True)
    return devices[0]
