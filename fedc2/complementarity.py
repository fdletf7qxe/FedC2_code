"""Class-wise coverage and reliability signals used by FedC2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _client_coverage(
    bags: Iterable[Mapping[str, object]], num_classes: int
) -> np.ndarray:
    weighted = np.zeros(num_classes, dtype=np.float64)
    sample_count = 0
    for bag in bags:
        size = int(bag["size"])
        proportion = np.asarray(bag["proportion"], dtype=np.float64)
        if size <= 0 or proportion.shape != (num_classes,):
            raise ValueError("invalid bag metadata")
        if np.any(proportion < 0) or not np.isclose(proportion.sum(), 1.0, atol=1e-6):
            raise ValueError("invalid bag proportions")
        weighted += size * proportion
        sample_count += size
    if sample_count <= 0:
        raise ValueError("each client must contain at least one bag")
    return weighted / float(sample_count)


def build_class_coverage_gains(
    bag_metadata: Mapping[int, Iterable[Mapping[str, object]]],
    num_classes: int,
    epsilon: float = 1e-12,
) -> tuple[dict[int, np.ndarray], dict[int, dict[int, np.ndarray]]]:
    """Compute sample-weighted coverage and Eq. (7) for all client pairs."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    coverage = {
        client_id: _client_coverage(bags, num_classes)
        for client_id, bags in bag_metadata.items()
    }
    gains: dict[int, dict[int, np.ndarray]] = {}
    for target_id, target_coverage in coverage.items():
        gains[target_id] = {}
        denominator = np.maximum(1.0 - target_coverage, epsilon)
        for source_id, source_coverage in coverage.items():
            if source_id == target_id:
                gains[target_id][source_id] = np.zeros(num_classes, dtype=np.float64)
            else:
                gains[target_id][source_id] = np.clip(
                    np.maximum(source_coverage - target_coverage, 0.0) / denominator,
                    0.0,
                    1.0,
                )
    return coverage, gains


def estimate_source_reliability(
    model: nn.Module,
    bags: Iterable[Mapping[str, torch.Tensor]],
    num_classes: int,
    device: torch.device | str,
    bag_limit: int = 16,
) -> np.ndarray:
    """Estimate the source-local reliability vector from Eqs. (8)-(9).

    Each bag must contain `images` and `proportion`. Instance labels are neither
    accepted nor used. The caller is responsible for applying evaluation-time
    preprocessing before passing images.
    """
    selected = list(bags)[:bag_limit]
    if not selected:
        return np.zeros(num_classes, dtype=np.float64)
    if bag_limit <= 0:
        raise ValueError("bag_limit must be positive")

    was_training = model.training
    model = model.to(device)
    model.eval()
    weighted_error = np.zeros(num_classes, dtype=np.float64)
    total_samples = 0
    with torch.no_grad():
        for bag in selected:
            images = bag["images"].to(device)
            proportion = bag["proportion"].detach().cpu().numpy()
            if proportion.shape != (num_classes,):
                raise ValueError("bag proportion has the wrong number of classes")
            forward_logits = getattr(model, "forward_logits", model.forward)
            predicted = torch.softmax(forward_logits(images), dim=1).mean(dim=0)
            bag_size = int(images.shape[0])
            weighted_error += bag_size * np.abs(
                predicted.detach().cpu().numpy() - proportion
            )
            total_samples += bag_size
    model.train(was_training)
    mean_error = weighted_error / float(total_samples)
    return np.clip(1.0 - mean_error, 0.0, 1.0)


def build_complementarity_scores(
    coverage_gains: Mapping[int, Mapping[int, Sequence[float]]],
    source_reliability: Mapping[int, Sequence[float]],
) -> dict[int, dict[int, np.ndarray]]:
    """Multiply target-relative coverage by source reliability as in Eq. (10)."""
    result: dict[int, dict[int, np.ndarray]] = {}
    for target_id, sources in coverage_gains.items():
        result[target_id] = {}
        for source_id, gain in sources.items():
            gain_array = np.clip(np.asarray(gain, dtype=np.float64), 0.0, 1.0)
            if source_id == target_id:
                result[target_id][source_id] = np.zeros_like(gain_array)
                continue
            reliability = np.clip(
                np.asarray(source_reliability[source_id], dtype=np.float64),
                0.0,
                1.0,
            )
            if reliability.shape != gain_array.shape:
                raise ValueError("coverage and reliability shapes must match")
            result[target_id][source_id] = gain_array * reliability
    return result

