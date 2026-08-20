"""AIra-v2 event-triggered language substrate primitives."""

from .calibration import ReliabilityThreshold, calibrate_reliability_threshold
from .memory import BoundedAssociativeMemory, MemoryHit
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
    "BoundedAssociativeMemory",
    "CompactShelfLevel",
    "MemoryHit",
    "ReliabilityThreshold",
    "ShelfEvaluation",
    "ShelfPrediction",
    "ShelfRoutes",
    "backprop_gradients",
    "build_compact_shelf",
    "calibrate_reliability_threshold",
    "evaluate_hierarchical_shelf",
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
]
