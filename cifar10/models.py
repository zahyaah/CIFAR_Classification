"""Baseline CNN and its residual variant.

Both models share the same layout: three stages of two 3x3 convolutions
(32, 64, 128 channels), max-pooling between stages, global average pooling
and a dropout + linear head. ``ResidualCNN`` wraps each stage's two
convolutions in a ``ResidualBlock``, so the shortcuts are the only difference.
See docs/decisions/ADR-002-residual-block.md.
"""
from torch import Tensor, nn

STAGE_CHANNELS = (32, 64, 128)


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    # bias=False: BatchNorm's shift makes a conv bias redundant.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def classification_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )


class BaselineCNN(nn.Module):
    """Six conv-BN-ReLU layers in three stages, then a dropout + linear head."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for i, channels in enumerate(STAGE_CHANNELS):
            if i > 0:
                layers.append(nn.MaxPool2d(2))
            layers += [conv_bn_relu(in_channels, channels), conv_bn_relu(channels, channels)]
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.head = classification_head(in_channels, num_classes, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    """Basic ResNet block: two 3x3 conv-BN layers plus a shortcut, then ReLU.

    The shortcut is the identity when input and output shapes match, otherwise
    a 1x1 conv + BatchNorm projection (option B in He et al., 2015,
    https://arxiv.org/abs/1512.03385).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResidualCNN(nn.Module):
    """``BaselineCNN`` with each stage's conv pair replaced by a ``ResidualBlock``."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for i, channels in enumerate(STAGE_CHANNELS):
            if i > 0:
                layers.append(nn.MaxPool2d(2))
            layers.append(ResidualBlock(in_channels, channels))
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.head = classification_head(in_channels, num_classes, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


MODELS = {"baseline": BaselineCNN, "resnet": ResidualCNN}


def build_model(name: str, num_classes: int = 10, dropout: float = 0.3) -> nn.Module:
    """Instantiate a model by name: ``"baseline"`` or ``"resnet"``."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {sorted(MODELS)}.")
    return MODELS[name](num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
