"""AIra-v2 event-triggered language substrate primitives."""

from .bridge import BridgedShelfPrediction, ByteBPEBridge
from .calibration import ReliabilityThreshold, calibrate_reliability_threshold
from .cascade import (
    AIraCascade,
    ByteEventConfig,
    ByteEventGenerationResult,
    CognitiveCascadeResult,
    generate_byte_events,
    utf8_allowed_next_bytes,
)
from .event_core import (
    AttentionByteEventLM,
    ByteEventLM,
    ConvByteEventLM,
    EventContextLM,
)
from .memory import (
    BoundedAssociativeMemory,
    EpisodicFactStore,
    MemoryHit,
    StructuredKeyEncoder,
)
from .mixture import normalized_shelf_neural_mixture, shelf_distribution
from .pc_alm import (
    backprop_gradients,
    gradient_cosine,
    local_augmented_gradients,
    minimum_layer_cosine,
)
from .residual import residual_training_weights
from .trigger import (
    CompactShelfLevel,
    ShelfEvaluation,
    ShelfPrediction,
    ShelfRoutes,
    build_compact_shelf,
    evaluate_hierarchical_shelf,
    load_compact_shelf,
    lookup_compact_level,
    predict_shelf_next,
    route_hierarchical_shelf,
    save_compact_shelf,
)

__all__ = [
    "AIraCascade",
    "AttentionByteEventLM",
    "BoundedAssociativeMemory",
    "BridgedShelfPrediction",
    "ByteBPEBridge",
    "ByteEventConfig",
    "ByteEventGenerationResult",
    "ByteEventLM",
    "CognitiveCascadeResult",
    "CompactShelfLevel",
    "ConvByteEventLM",
    "EpisodicFactStore",
    "EventContextLM",
    "MemoryHit",
    "ReliabilityThreshold",
    "ShelfEvaluation",
    "ShelfPrediction",
    "ShelfRoutes",
    "StructuredKeyEncoder",
    "backprop_gradients",
    "build_compact_shelf",
    "calibrate_reliability_threshold",
    "evaluate_hierarchical_shelf",
    "generate_byte_events",
    "gradient_cosine",
    "load_compact_shelf",
    "local_augmented_gradients",
    "lookup_compact_level",
    "minimum_layer_cosine",
    "normalized_shelf_neural_mixture",
    "predict_shelf_next",
    "residual_training_weights",
    "route_hierarchical_shelf",
    "save_compact_shelf",
    "shelf_distribution",
    "utf8_allowed_next_bytes",
]
