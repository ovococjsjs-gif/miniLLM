"""Contamination-resistant synthetic probes for architecture ablations."""

from .associative_recall import RecallBatch, evaluate_recall, generate_recall_batch

__all__ = ["RecallBatch", "evaluate_recall", "generate_recall_batch"]
