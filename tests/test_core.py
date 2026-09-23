import unittest

import numpy as np
import torch
from torch import nn

from fedc2.aggregation import (
    aggregate_fedc2_models,
    build_compatibility,
    compatibility_weights,
    consolidate_global_model,
    proportion_set_distance,
    select_collaborators,
)
from fedc2.complementarity import (
    build_class_coverage_gains,
    build_complementarity_scores,
)
from fedc2.local_objective import dllp_bag_loss


class TinyModel(nn.Module):
    def __init__(self, feature: float, rows: list[list[float]]):
        super().__init__()
        self.feature = nn.Linear(1, 1, bias=False)
        self.classifier = nn.Linear(1, 2)
        with torch.no_grad():
            self.feature.weight.fill_(feature)
            self.classifier.weight.copy_(torch.tensor(rows, dtype=torch.float32))
            self.classifier.bias.zero_()

    def get_classifier_parameter_names(self):
        return "classifier.weight", "classifier.bias"


class FedC2CoreTests(unittest.TestCase):
    def test_set_distance_is_symmetric(self):
        first = [[0.9, 0.1], [0.7, 0.3]]
        second = [[0.2, 0.8]]
        self.assertAlmostEqual(
            proportion_set_distance(first, second),
            proportion_set_distance(second, first),
        )

    def test_compatibility_and_top_h(self):
        sets = {0: [[0.9, 0.1]], 1: [[0.8, 0.2]], 2: [[0.1, 0.9]]}
        _, compatibility, bandwidth = build_compatibility(sets)
        self.assertGreater(bandwidth, 0)
        self.assertEqual(select_collaborators(compatibility, 1)[0], [1])

    def test_sample_count_weighted_base(self):
        weights = compatibility_weights(
            0,
            [1],
            {0: {0: 1.0, 1: 0.5}},
            {0: 10, 1: 20},
        )
        self.assertAlmostEqual(weights[0], 0.5)
        self.assertAlmostEqual(weights[1], 0.5)

    def test_coverage_and_reliability_product(self):
        metadata = {
            0: [{"size": 10, "proportion": [0.9, 0.1]}],
            1: [{"size": 10, "proportion": [0.1, 0.9]}],
        }
        _, gains = build_class_coverage_gains(metadata, 2)
        scores = build_complementarity_scores(
            gains, {0: np.array([0.8, 0.7]), 1: np.array([0.6, 0.9])}
        )
        self.assertAlmostEqual(scores[0][1][0], 0.0)
        self.assertAlmostEqual(scores[0][1][1], 0.8)

    def test_refinement_changes_only_supported_classifier_rows(self):
        models = {
            0: TinyModel(1.0, [[1.0], [1.0]]),
            1: TinyModel(3.0, [[3.0], [3.0]]),
        }
        states, _ = aggregate_fedc2_models(
            models=models,
            collaborators={0: [1], 1: [0]},
            compatibility={0: {0: 1.0, 1: 1.0}, 1: {0: 1.0, 1: 1.0}},
            sample_counts={0: 1, 1: 1},
            complementarity={
                0: {1: np.array([0.25, 0.0])},
                1: {0: np.array([0.0, 0.0])},
            },
            strength=0.2,
            cap=0.25,
        )
        self.assertTrue(torch.equal(states[0]["feature.weight"], torch.tensor([[2.0]])))
        self.assertAlmostEqual(states[0]["classifier.weight"][0].item(), 2.2, places=6)
        self.assertAlmostEqual(states[0]["classifier.weight"][1].item(), 2.0, places=6)

    def test_global_consolidation_is_sample_weighted(self):
        models = {
            0: TinyModel(1.0, [[1.0], [1.0]]),
            1: TinyModel(3.0, [[3.0], [3.0]]),
        }
        state = consolidate_global_model(models, {0: 1, 1: 3})
        self.assertAlmostEqual(state["feature.weight"].item(), 2.5)

    def test_dllp_loss_uses_bag_fraction(self):
        logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        target = torch.tensor([1.0, 0.0])
        full = dllp_bag_loss(logits, target, bag_size=2, client_sample_count=2)
        half = dllp_bag_loss(logits, target, bag_size=2, client_sample_count=4)
        self.assertAlmostEqual(half.item(), full.item() / 2.0)


if __name__ == "__main__":
    unittest.main()

