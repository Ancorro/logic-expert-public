"""Baseline model: backbone + task head only, no logic stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel

from .logic_llama_model import _detect_causal


@dataclass
class BaselineModelOutput:
    """Minimal baseline output container.

    Attributes:
        logits: Classification logits from the baseline head.
    """
    logits: torch.Tensor


class BaselineModel(nn.Module):
    """Backbone-only classifier used as the non-logic comparison model.

    This wrapper intentionally omits logic projection, routing, and fusion modules.
    It runs the backbone, pools a token representation, and applies a linear task
    head. This makes it a clean baseline against logic-augmented variants.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        backbone: Optional[nn.Module] = None,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        if backbone is None and model_name is None:
            raise ValueError("Provide either model_name or backbone")
        if backbone is None:
            backbone = AutoModel.from_pretrained(model_name)
        self.backbone = backbone
        self._causal = _detect_causal(self.backbone.config)
        self._head_device: Optional[str] = None
        hidden_dim = int(self.backbone.config.hidden_size)
        self.task_head = nn.Linear(hidden_dim, num_labels)

    def enable_model_parallel(self, devices: list[str]) -> None:
        """Split backbone layers across ``devices``; the head stays on ``devices[0]``."""
        from .model_parallel import dispatch_backbone

        self._head_device = dispatch_backbone(self.backbone, devices)
        self.task_head.to(self._head_device)

    def _align_task_head_to(self, ref: torch.Tensor) -> None:
        """Keep task head on the same runtime device+dtype as backbone outputs."""
        if self.task_head.weight.device == ref.device and self.task_head.weight.dtype == ref.dtype:
            return
        self.task_head.to(device=ref.device, dtype=ref.dtype)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters so only the task head remains trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def enable_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        """Attach PEFT LoRA adapters to the backbone for parameter-efficient tuning.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA scaling factor used by PEFT ``LoraConfig``.
            dropout: LoRA dropout probability.
            target_modules: Backbone module names to inject adapters into.

        Notes:
            ``lora_alpha`` is specific to LoRA adapters and not related to fusion
            ``alpha_init`` used by logic-stream models.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "PEFT is required for LoRA. Install with: pip install peft"
            ) from exc

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BaselineModelOutput:
        """Run baseline forward pass and return logits.

        Pools the last non-padding token for causal/decoder backbones (as
        ``LogicLlamaModel`` does) and position 0 for encoders. Pooling position 0 on a
        causal model sees only the BOS token, which makes the classifier input-blind.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if self._head_device is not None:
            last_hidden = last_hidden.to(self._head_device)
        if self._causal:
            seq_lens = attention_mask.to(last_hidden.device).sum(dim=1) - 1
            hidden = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]
        else:
            hidden = last_hidden[:, 0, :]
        self._align_task_head_to(hidden)
        logits = self.task_head(hidden)
        return BaselineModelOutput(logits=logits)
