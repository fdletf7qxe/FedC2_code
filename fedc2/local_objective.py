"""Bag-level DLLP objective used for FedC2 local optimization."""

from collections.abc import Iterable

import torch


def dllp_bag_loss(
    logits: torch.Tensor,
    target_proportion: torch.Tensor,
    bag_size: int,
    client_sample_count: int,
) -> torch.Tensor:
    """Return the sample-weighted bag proportion cross-entropy.

    This implements Eq. (15) for one bag. Summing this value over all local
    bags gives the client objective in the paper.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [bag_size, num_classes]")
    if target_proportion.ndim != 1 or target_proportion.numel() != logits.shape[1]:
        raise ValueError("target_proportion must match the classifier output")
    if bag_size != logits.shape[0] or bag_size <= 0:
        raise ValueError("bag_size must match the number of logits")
    if client_sample_count <= 0:
        raise ValueError("client_sample_count must be positive")

    predicted_proportion = torch.softmax(logits, dim=1).mean(dim=0)
    cross_entropy = -(
        target_proportion
        * torch.log(predicted_proportion.clamp_min(1e-8))
    ).sum()
    bag_weight = round(float(bag_size) / float(client_sample_count), 6)
    return bag_weight * cross_entropy


def dllp_step_loss(losses: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum independently computed bag losses for one optimization step."""
    losses = list(losses)
    if not losses:
        raise ValueError("at least one bag loss is required")
    return torch.stack(losses).sum()
