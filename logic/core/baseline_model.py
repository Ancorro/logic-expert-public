"""Baseline model: backbone + task head only, no logic stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel


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
        hidden_dim = int(self.backbone.config.hidden_size)
        self.task_head = nn.Linear(hidden_dim, num_labels)

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

        Uses the first-token representation from ``last_hidden_state`` as pooled
        features before the classifier head.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        self._align_task_head_to(hidden)
        logits = self.task_head(hidden)
        return BaselineModelOutput(logits=logits)
