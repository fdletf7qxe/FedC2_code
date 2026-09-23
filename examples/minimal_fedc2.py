"""Small synthetic example of one FedC2 server aggregation step."""

import numpy as np
import torch
from torch import nn

from fedc2 import (
    aggregate_fedc2_models,
    build_class_coverage_gains,
    build_compatibility,
    build_complementarity_scores,
    select_collaborators,
)


class TinyClassifier(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.classifier.weight.fill_(value)
            self.classifier.bias.fill_(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))

    def get_classifier_parameter_names(self) -> tuple[str, str]:
        return "classifier.weight", "classifier.bias"


def main() -> None:
    bag_metadata = {
        0: [{"size": 8, "proportion": [0.75, 0.25]}],
        1: [{"size": 8, "proportion": [0.25, 0.75]}],
        2: [{"size": 8, "proportion": [0.50, 0.50]}],
    }
    proportion_sets = {
        client_id: [bag["proportion"] for bag in bags]
        for client_id, bags in bag_metadata.items()
    }
    _, compatibility, _ = build_compatibility(proportion_sets)
    collaborators = select_collaborators(compatibility, count=2)
    _, coverage_gains = build_class_coverage_gains(bag_metadata, num_classes=2)
    reliability = {
        0: np.array([0.90, 0.80]),
        1: np.array([0.85, 0.95]),
        2: np.array([0.90, 0.90]),
    }
    complementarity = build_complementarity_scores(coverage_gains, reliability)
    models = {client_id: TinyClassifier(float(client_id + 1)) for client_id in range(3)}
    states, _ = aggregate_fedc2_models(
        models=models,
        collaborators=collaborators,
        compatibility=compatibility,
        sample_counts={0: 8, 1: 8, 2: 8},
        complementarity=complementarity,
    )
    print(states[0]["classifier.weight"])


if __name__ == "__main__":
    main()

