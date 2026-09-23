"""Minimal reference implementation of FedC2."""

from .aggregation import (
    aggregate_fedc2_models,
    build_compatibility,
    consolidate_global_model,
    select_collaborators,
)
from .complementarity import (
    build_class_coverage_gains,
    build_complementarity_scores,
    estimate_source_reliability,
)
from .local_objective import dllp_bag_loss, dllp_step_loss
from .model import FedC2ResNet18

__all__ = [
    "FedC2ResNet18",
    "aggregate_fedc2_models",
    "build_class_coverage_gains",
    "build_compatibility",
    "build_complementarity_scores",
    "consolidate_global_model",
    "dllp_bag_loss",
    "dllp_step_loss",
    "estimate_source_reliability",
    "select_collaborators",
]

