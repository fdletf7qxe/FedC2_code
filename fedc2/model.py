"""Checkpoint-compatible ResNet-18 used in the FedC2 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(outputs + residual)


class _ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize()

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(self.inplanes, channels, stride)]
        self.inplanes = channels
        layers.extend(_BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        return torch.flatten(self.avgpool(outputs), 1)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class FedC2ResNet18(nn.Module):
    """Formal FedC2 backbone with the original state-dict layout."""

    def __init__(self, num_classes: int, projection_dim: int = 128):
        super().__init__()
        self.backbone = _ResNet18Backbone(num_classes)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(inputs)

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(self.features(inputs))

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(features), dim=1)

    def forward_with_projection(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        return self.backbone.classifier(features), self.project_features(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(inputs)

    def get_classifier_parameter_names(self) -> tuple[str, str]:
        return "backbone.fc.weight", "backbone.fc.bias"

