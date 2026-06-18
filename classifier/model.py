"""EfficientNet-B0 architecture for the pizza/steak/sushi classifier.

Mirrors ``torchvision.models.efficientnet_b0`` with a classifier head adapted to
3 classes. It is aligned exactly with the weights in
``pizza_steak_sushi_classification_model.pth`` (state_dict):
``features.0``-``features.8`` plus ``classifier.1 = Linear(1280 -> 3)``.
"""
import torchvision
from torch import nn


def create_model(num_classes: int = 3) -> nn.Module:
    """Build the empty EfficientNet-B0 skeleton (no ImageNet download).

    ``weights=None`` only initialises the architecture; the trained weights are
    loaded afterwards in :func:`classifier.inference.get_model`.
    """
    model = torchvision.models.efficientnet_b0(weights=None)
    # Head adapted to 3 classes as in training (Dropout stays at index 0 so the
    # "classifier.1.*" key matches the checkpoint).
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return model
