from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel

from .fusion import FusionMLP, LinearFusionBridge
from .logic_cross_attn import LayerwiseLogicCrossAttention
from .logic_projection import LogicProjection
from .logic_stream import LogicStream

# Decoder-only (causal) model types — pool from last non-padding token
_CAUSAL_MODEL_TYPES = {
    "llama", "mistral", "falcon", "gpt2", "gpt_neo", "gpt_neox",
    "opt", "bloom", "phi", "gemma", "qwen2", "mixtral",
}


def _detect_causal(config) -> bool:
    """Infer whether pooling should follow decoder-style (causal) behavior."""
    return (
        getattr(config, "model_type", "") in _CAUSAL_MODEL_TYPES
        or getattr(config, "is_decoder", False)
    )


@dataclass
class LogicModelOutput:
    """Output container for logic-augmented model forward pass.

    Attributes:
        fused_hidden: Token-level fused representation ``[B, S, H]``.
        logits: Task logits (or ``None`` for headless scenarios).
        routing_history: Per-layer routing tensors for diagnostics.
        cross_attn_history: Per-layer attention probabilities for diagnostics.
    """
    fused_hidden: torch.Tensor
    logits: Optional[torch.Tensor]
    routing_history: list[torch.Tensor]
    cross_attn_history: list[torch.Tensor]


class LogicLlamaModel(nn.Module):
    """
    Logic-augmented wrapper around an AutoModel-compatible transformer.

    Architecture overview:
        1) Backbone produces hidden states for all transformer layers.
        2) Each layer hidden state is projected from ``H`` to logic width ``L``.
        3) ``LogicStream`` processes projected states sequentially to produce a final
           per-example logic state plus routing diagnostics.
        4) Final backbone layer state is fused with broadcast logic state via
           ``FusionMLP``.
        5) A normalized pooled representation is fed to a task head.

    Despite the class name, this wrapper supports any model compatible with
    ``transformers.AutoModel`` that returns hidden states.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        backbone: Optional[nn.Module] = None,
        stream_logic_during_backbone: bool = True,
        logic_dim: int = 128,
        num_gates: int = 16,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.0,
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
        skip_projection_when_dims_match: bool = False,
        fusion_mode: str = "linear_bridge",
        fusion_hidden_dim: int = 512,
        alpha_init: float = 0.01,
        learn_fusion_alpha: bool = True,
        num_labels: int = 2,
    ) -> None:
        super().__init__()

        if int(cross_attn_heads) < 1:
            raise ValueError("cross_attn_heads must be >= 1")

        if backbone is None and model_name is None:
            raise ValueError("Provide either model_name or backbone")
        if backbone is None:
            backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = backbone
        self._causal = _detect_causal(self.backbone.config)

        hidden_dim = int(self.backbone.config.hidden_size)
        num_layers = int(self.backbone.config.num_hidden_layers)

        self.logic_projection = LogicProjection(
            hidden_dim,
            logic_dim,
            skip_learned_mapping_when_dims_match=skip_projection_when_dims_match,
        )
        self.logic_stream = LogicStream(
            num_layers=num_layers,
            logic_dim=logic_dim,
            num_gates=num_gates,
            routing_top_k=routing_top_k,
            routing_train_mode=routing_train_mode,
            routing_train_cutoff=routing_train_cutoff,
            routing_temperature=routing_temperature,
            gate_mode=gate_mode,
            gate_operator=gate_operator,
            gate_capacity=gate_capacity,
            gate_window_axis=gate_window_axis,
            pre_routing_mlp=pre_routing_mlp,
            use_no_gate_stream=use_no_gate_stream,
            no_gate_update_type=no_gate_update_type,
            no_gate_update_hidden_dim=no_gate_update_hidden_dim,
            no_gate_match_logic_params=no_gate_match_logic_params,
        )
        self.cross_attn = nn.ModuleList(
            [
                LayerwiseLogicCrossAttention(
                    logic_dim=logic_dim,
                    hidden_dim=hidden_dim,
                    num_heads=int(cross_attn_heads),
                    dropout=cross_attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        if fusion_mode == "linear_bridge":
            self.fusion = LinearFusionBridge(
                hidden_dim=hidden_dim,
                logic_dim=logic_dim,
                alpha_init=alpha_init,
                learn_fusion_alpha=learn_fusion_alpha,
            )
        elif fusion_mode == "mlp":
            self.fusion = FusionMLP(
                hidden_dim=hidden_dim,
                logic_dim=logic_dim,
                fusion_hidden_dim=fusion_hidden_dim,
                alpha_init=alpha_init,
                learn_fusion_alpha=learn_fusion_alpha,
            )
        else:
            raise ValueError("fusion_mode must be one of {'linear_bridge', 'mlp'}")

        self.fusion_mode = fusion_mode
        if not stream_logic_during_backbone:
            raise ValueError(
                "stream_logic_during_backbone=False is legacy and no longer supported; streaming-only path is canonical"
            )

        # Streaming-only canonical path.
        self._stream_logic_during_backbone = True
        self._backbone_layers = self._infer_backbone_layer_stack()
        self.backbone.config.output_hidden_states = False

        if self._backbone_layers is None:
            raise RuntimeError("Backbone layer stack could not be discovered for streaming logic")
        if len(self._backbone_layers) != self.logic_stream.num_layers:
            raise RuntimeError(
                f"Layer-count mismatch: discovered {len(self._backbone_layers)} backbone layers but logic_stream expects {self.logic_stream.num_layers}"
            )

        # Match Llama-style stabilization before the classifier projection.
        if hasattr(nn, "RMSNorm"):
            self.pre_head_norm = nn.RMSNorm(hidden_dim)
        else:
            self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.task_head = nn.Linear(hidden_dim, num_labels)

    def _infer_backbone_layer_stack(self) -> Optional[list[nn.Module]]:
        """Find the ordered transformer block stack for common HF backbone layouts."""
        candidates: list[object] = []
        if hasattr(self.backbone, "model"):
            candidates.append(self.backbone.model)
        candidates.append(self.backbone)

        for owner in candidates:
            layers = getattr(owner, "layers", None)
            if isinstance(layers, nn.ModuleList):
                return list(layers)

            encoder = getattr(owner, "encoder", None)
            if encoder is not None:
                enc_layers = getattr(encoder, "layer", None)
                if isinstance(enc_layers, nn.ModuleList):
                    return list(enc_layers)

            transformer = getattr(owner, "transformer", None)
            if transformer is not None:
                tr_layers = getattr(transformer, "h", None)
                if isinstance(tr_layers, nn.ModuleList):
                    return list(tr_layers)

        return None

    @staticmethod
    def _extract_hidden_from_layer_output(layer_output: torch.Tensor | tuple | list) -> torch.Tensor:
        """Normalize transformer-layer outputs to hidden states [B,S,H]."""
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        if isinstance(layer_output, (tuple, list)) and layer_output and isinstance(layer_output[0], torch.Tensor):
            return layer_output[0]
        raise RuntimeError("Unsupported backbone layer output format for streaming logic")

    def _align_logic_modules_to(self, ref: torch.Tensor) -> None:
        """Keep logic/head modules on the same runtime device+dtype as backbone outputs."""
        target_device = ref.device
        target_dtype = ref.dtype

        def _module_aligned(module: nn.Module) -> bool:
            param = next(module.parameters(), None)
            if param is None:
                return True
            return param.device == target_device and param.dtype == target_dtype

        if all(
            _module_aligned(module)
            for module in (
                self.logic_projection,
                self.logic_stream,
                self.cross_attn,
                self.fusion,
                self.pre_head_norm,
                self.task_head,
            )
        ):
            return

        self.logic_projection.to(device=target_device, dtype=target_dtype)
        self.logic_stream.to(device=target_device, dtype=target_dtype)
        self.cross_attn.to(device=target_device, dtype=target_dtype)
        self.fusion.to(device=target_device, dtype=target_dtype)
        self.pre_head_norm.to(device=target_device, dtype=target_dtype)
        self.task_head.to(device=target_device, dtype=target_dtype)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters while keeping logic/head modules trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def enable_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        """Attach PEFT LoRA adapters to the backbone.

        Args:
            r: LoRA rank.
            lora_alpha: LoRA scaling factor used by PEFT ``LoraConfig``.
            dropout: LoRA dropout probability.
            target_modules: Backbone module names to inject adapters into.

        Notes:
            This LoRA scaling is unrelated to fusion alpha. Fusion alpha is configured
            via ``alpha_init`` in ``FusionMLP`` and logged as ``train/fusion_alpha``.
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> LogicModelOutput:
        """Run the logic-augmented forward pipeline.

        Notes:
            - In streaming mode, logic updates run layer-by-layer during backbone forward
              via per-layer hooks to reduce hidden-state retention overhead.
            - In fallback mode, the model consumes ``output_hidden_states`` after the
              backbone forward.
            - Pooling strategy depends on model type: decoder/causal models use the
              last non-padding token, while encoder models use position 0.
        """
        state_holder: dict[str, Optional[torch.Tensor]] = {"logic_state": None}
        routing_history_buffer: list[Optional[torch.Tensor]] = [None] * self.logic_stream.num_layers
        cross_attn_history_buffer: list[Optional[torch.Tensor]] = [None] * self.logic_stream.num_layers

        token_mask = attention_mask.to(dtype=torch.bool)

        def _make_hook(layer_idx: int):
            def _hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor | tuple | list):
                hidden = self._extract_hidden_from_layer_output(output)
                if hidden.ndim != 3:
                    raise RuntimeError(
                        f"Expected hidden states [B,S,H] from backbone layer {layer_idx}, got {tuple(hidden.shape)}"
                    )

                if state_holder["logic_state"] is None:
                    self._align_logic_modules_to(hidden)
                    init_state = self.logic_projection(hidden)
                    init_state = init_state * token_mask.unsqueeze(-1).to(dtype=init_state.dtype)
                    state_holder["logic_state"] = init_state

                if self.logic_stream.use_no_gate_stream:
                    projected_logic = self.logic_projection(hidden)
                    projected_logic = projected_logic * token_mask.unsqueeze(-1).to(dtype=projected_logic.dtype)
                    logic_state, routing_weights = self.logic_stream.layers[layer_idx](
                        projected_logic,
                        state_holder["logic_state"],
                        inference=not self.training,
                        token_mask=token_mask,
                    )
                    state_holder["logic_state"] = logic_state
                    routing_history_buffer[layer_idx] = routing_weights
                else:
                    attended_logic, attn_probs = self.cross_attn[layer_idx](
                        state_holder["logic_state"],
                        hidden,
                        token_mask=token_mask,
                    )
                    logic_state, routing_weights = self.logic_stream.layers[layer_idx](
                        attended_logic,
                        attended_logic,
                        inference=not self.training,
                        token_mask=token_mask,
                    )
                    state_holder["logic_state"] = logic_state
                    routing_history_buffer[layer_idx] = routing_weights
                    cross_attn_history_buffer[layer_idx] = attn_probs
                return output

            return _hook

        handles = [
            layer.register_forward_hook(_make_hook(layer_idx))
            for layer_idx, layer in enumerate(self._backbone_layers)
        ]
        try:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for handle in handles:
                handle.remove()

        llm_hidden = outputs.last_hidden_state
        logic_state = state_holder["logic_state"]
        if logic_state is None:
            raise RuntimeError("Streaming logic hooks did not produce a logic state")

        missing = [idx for idx, value in enumerate(routing_history_buffer) if value is None]
        if missing:
            raise RuntimeError(f"Streaming logic hooks missed routing outputs for layers: {missing}")
        routing_history = [value for value in routing_history_buffer if value is not None]
        if self.logic_stream.use_no_gate_stream:
            cross_attn_history = []
        else:
            cross_missing = [idx for idx, value in enumerate(cross_attn_history_buffer) if value is None]
            if cross_missing:
                raise RuntimeError(f"Streaming hooks missed cross-attn outputs for layers: {cross_missing}")
            cross_attn_history = [value for value in cross_attn_history_buffer if value is not None]

        fused_hidden = self.fusion(llm_hidden, logic_state, token_mask=token_mask)

        if self._causal:
            # Causal/decoder models: pool from the last non-padding token per example
            seq_lens = attention_mask.to(fused_hidden.device).sum(dim=1) - 1  # [B] index of last real token
            pooled = fused_hidden[torch.arange(fused_hidden.size(0), device=fused_hidden.device), seq_lens]
        else:
            # Encoder models: use the [CLS] token at position 0
            pooled = fused_hidden[:, 0, :]
        pooled = self.pre_head_norm(pooled)
        logits = self.task_head(pooled)
        return LogicModelOutput(
            fused_hidden=fused_hidden,
            logits=logits,
            routing_history=routing_history,
            cross_attn_history=cross_attn_history,
        )
