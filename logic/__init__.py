"""Public entry points for logic-stream components.

This package re-exports the most commonly used modules so callers can import
core building blocks from ``logic`` directly. ``LogicLlamaModel`` is imported
lazily/optionally because lightweight environments may not have ``transformers``
installed.
"""

from .core.fusion import FusionMLP, LinearFusionBridge
from .core.logic_cross_attn import LayerwiseLogicCrossAttention
from .core.logic_layer import LogicLayer
from .core.logic_projection import LogicProjection
from .core.logic_stream import LogicStream
from .core.routing import RoutingModule

try:
    from .core.logic_llama_model import LogicLlamaModel
except Exception:  # transformers may be unavailable in lightweight envs
    LogicLlamaModel = None

__all__ = [
    "FusionMLP",
    "LinearFusionBridge",
    "LayerwiseLogicCrossAttention",
    "LogicLayer",
    "LogicProjection",
    "LogicStream",
    "RoutingModule",
]

if LogicLlamaModel is not None:
    __all__.append("LogicLlamaModel")
