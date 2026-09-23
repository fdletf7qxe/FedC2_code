"""Compatibility-guided and class-wise complementary aggregation for FedC2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two class-proportion vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("proportion vectors must have the same one-dimensional shape")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("proportions must be non-negative")
    if not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("proportion vectors must sum to one")
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint)


def proportion_set_distance(
    proportions_a: Sequence[Sequence[float]],
    proportions_b: Sequence[Sequence[float]],
) -> float:
    """Symmetric nearest-neighbor discrepancy from Eq. (3)."""
    matrix_a = np.asarray(proportions_a, dtype=np.float64)
    matrix_b = np.asarray(proportions_b, dtype=np.float64)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2 or matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError("proportion sets must be non-empty matrices with equal width")
    if matrix_a.shape[0] == 0 or matrix_b.shape[0] == 0:
        raise ValueError("proportion sets must be non-empty")

    distances = np.empty((matrix_a.shape[0], matrix_b.shape[0]), dtype=np.float64)
    for row_a in range(matrix_a.shape[0]):
        for row_b in range(matrix_b.shape[0]):
            distances[row_a, row_b] = js_divergence(
                matrix_a[row_a], matrix_b[row_b]
            )
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def build_compatibility(
    proportion_sets: Mapping[int, Sequence[Sequence[float]]],
    bandwidth: float | str = "mean_pairwise_distance",
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]], float]:
    """Return pairwise discrepancy, compatibility, and resolved bandwidth."""
    client_ids = sorted(proportion_sets)
    if len(client_ids) < 2:
        raise ValueError("at least two clients are required")
    distances = {client_id: {client_id: 0.0} for client_id in client_ids}
    non_self_distances: list[float] = []
    for offset, client_id in enumerate(client_ids):
        for other_id in client_ids[offset + 1 :]:
            distance = proportion_set_distance(
                proportion_sets[client_id], proportion_sets[other_id]
            )
            distances[client_id][other_id] = distance
            distances[other_id][client_id] = distance
            non_self_distances.append(distance)

    if bandwidth == "mean_pairwise_distance":
        resolved_bandwidth = float(np.mean(non_self_distances))
    else:
        resolved_bandwidth = float(bandwidth)
    if not np.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0:
        raise ValueError("compatibility bandwidth must be positive")

    compatibility = {client_id: {} for client_id in client_ids}
    for client_id in client_ids:
        for other_id in client_ids:
            compatibility[client_id][other_id] = (
                1.0
                if client_id == other_id
                else float(math.exp(-distances[client_id][other_id] / resolved_bandwidth))
            )
    return distances, compatibility, resolved_bandwidth


def select_collaborators(
    compatibility: Mapping[int, Mapping[int, float]], count: int
) -> dict[int, list[int]]:
    """Select the H highest-scoring non-self clients for every target."""
    if count < 0:
        raise ValueError("collaborator count must be non-negative")
    selected: dict[int, list[int]] = {}
    for client_id, row in compatibility.items():
        candidates = [
            (other_id, float(score))
            for other_id, score in row.items()
            if other_id != client_id
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected[client_id] = [client for client, _ in candidates[:count]]
    return selected


def compatibility_weights(
    target_id: int,
    collaborator_ids: Sequence[int],
    compatibility: Mapping[int, Mapping[int, float]],
    sample_counts: Mapping[int, int],
) -> dict[int, float]:
    """Compute Eq. (4) over the target and its selected collaborators."""
    source_ids = [target_id, *collaborator_ids]
    scores = {
        source_id: float(compatibility[target_id][source_id])
        * int(sample_counts[source_id])
        for source_id in source_ids
    }
    denominator = sum(scores.values())
    if denominator <= 0:
        raise ValueError("aggregation denominator must be positive")
    return {source_id: score / denominator for source_id, score in scores.items()}


def _snapshot_states(models: Mapping[int, nn.Module]) -> dict[int, dict[str, torch.Tensor]]:
    return {
        client_id: {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        for client_id, model in models.items()
    }


def _weighted_state(
    target_id: int,
    source_ids: Sequence[int],
    source_states: Mapping[int, Mapping[str, torch.Tensor]],
    weights: Mapping[int, float],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, target_value in source_states[target_id].items():
        if not torch.is_floating_point(target_value):
            result[name] = target_value.clone()
            continue
        value = torch.zeros_like(target_value)
        for source_id in source_ids:
            value.add_(source_states[source_id][name] * weights[source_id])
        result[name] = value
    return result


def _classifier_parameter_names(model: nn.Module) -> tuple[str, str]:
    getter = getattr(model, "get_classifier_parameter_names", None)
    if getter is None:
        raise TypeError("FedC2 models must expose get_classifier_parameter_names()")
    names = tuple(getter())
    if len(names) != 2:
        raise ValueError("classifier interface must return weight and bias names")
    return names[0], names[1]


def aggregate_fedc2_models(
    models: Mapping[int, nn.Module],
    collaborators: Mapping[int, Sequence[int]],
    compatibility: Mapping[int, Mapping[int, float]],
    sample_counts: Mapping[int, int],
    complementarity: Mapping[int, Mapping[int, Sequence[float]]],
    strength: float = 0.05,
    cap: float = 0.25,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[int, float]]]:
    """Apply Eqs. (4)-(5) and (11)-(14) to locally updated models.

    Non-classifier tensors use the compatibility-guided base. Classifier rows
    with no positive complementarity mass fall back exactly to the base state.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not 0.0 < cap <= 1.0:
        raise ValueError("cap must be in (0, 1]")

    source_states = _snapshot_states(models)
    aggregated: dict[int, dict[str, torch.Tensor]] = {}
    base_weights: dict[int, dict[int, float]] = {}

    for target_id in sorted(models):
        source_ids = [target_id, *collaborators[target_id]]
        alpha = compatibility_weights(
            target_id, collaborators[target_id], compatibility, sample_counts
        )
        base_weights[target_id] = alpha
        state = _weighted_state(target_id, source_ids, source_states, alpha)
        weight_name, bias_name = _classifier_parameter_names(models[target_id])
        if weight_name not in state or bias_name not in state:
            raise ValueError("classifier parameters are missing from model state")
        class_count = int(state[weight_name].shape[0])

        for class_id in range(class_count):
            capped = {
                source_id: (
                    0.0
                    if source_id == target_id
                    else min(
                        cap,
                        max(
                            0.0,
                            float(
                                complementarity.get(target_id, {})
                                .get(source_id, np.zeros(class_count))[class_id]
                            ),
                        ),
                    )
                )
                for source_id in source_ids
            }
            normalizer = sum(alpha[source_id] * capped[source_id] for source_id in source_ids)
            if strength == 0.0 or normalizer <= 0.0:
                continue
            beta = {
                source_id: alpha[source_id] * capped[source_id] / normalizer
                for source_id in source_ids
            }
            refined = {
                source_id: (1.0 - strength) * alpha[source_id]
                + strength * beta[source_id]
                for source_id in source_ids
            }
            state[weight_name][class_id].zero_()
            state[bias_name][class_id].zero_()
            for source_id in source_ids:
                state[weight_name][class_id].add_(
                    source_states[source_id][weight_name][class_id]
                    * refined[source_id]
                )
                state[bias_name][class_id].add_(
                    source_states[source_id][bias_name][class_id]
                    * refined[source_id]
                )
        aggregated[target_id] = state
    return aggregated, base_weights


def consolidate_global_model(
    models: Mapping[int, nn.Module], sample_counts: Mapping[int, int]
) -> dict[str, torch.Tensor]:
    """Sample-size-weighted consolidation from Eq. (16)."""
    client_ids = sorted(models)
    total_samples = sum(int(sample_counts[client_id]) for client_id in client_ids)
    if total_samples <= 0:
        raise ValueError("total sample count must be positive")
    weights = {
        client_id: int(sample_counts[client_id]) / float(total_samples)
        for client_id in client_ids
    }
    return _weighted_state(client_ids[0], client_ids, _snapshot_states(models), weights)

