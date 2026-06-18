"""Model loading and image prediction.

The model is loaded exactly once (lazily) and reused across all requests.
Inference runs on the CPU unless a CUDA GPU is available.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from .model import create_model

# Order as assigned by torchvision.datasets.ImageFolder (alphabetical).
CLASS_NAMES = ["pizza", "steak", "sushi"]

# Per-class display metadata (label, emoji, bar colour).
CLASS_META = {
    "pizza": {"label": "Pizza", "emoji": "🍕", "color": "#ef4444"},
    "steak": {"label": "Steak", "emoji": "🥩", "color": "#b45309"},
    "sushi": {"label": "Sushi", "emoji": "🍣", "color": "#fb7185"},
}

MODEL_FILENAME = "pizza_steak_sushi_classification_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Exactly the preprocessing of the ImageNet weights EfficientNet-B0 was
# pretrained on: Resize(256) -> CenterCrop(224) -> [0,1] -> Normalize
# (ImageNet mean/std). Only returns the transform, does NOT download weights.
_transform = EfficientNet_B0_Weights.DEFAULT.transforms()

_model: torch.nn.Module | None = None


def _model_path() -> Path:
    """Path to the .pth file in the project root."""
    from django.conf import settings
    return Path(settings.BASE_DIR) / MODEL_FILENAME


def get_model() -> torch.nn.Module:
    """Load the model on first call and cache the instance."""
    global _model
    if _model is None:
        model = create_model(num_classes=len(CLASS_NAMES))
        state_dict = torch.load(_model_path(), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def predict(image: Image.Image) -> list[dict]:
    """Classify a PIL image.

    Returns a list sorted by descending probability:
    ``[{"name", "label", "emoji", "color", "prob", "percent", "is_top"}, ...]``.
    """
    model = get_model()

    image_tensor = _transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    results = []
    for name, prob in zip(CLASS_NAMES, probs.tolist()):
        meta = CLASS_META[name]
        results.append({
            "name": name,
            "label": meta["label"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "prob": prob,
            "percent": round(prob * 100, 1),
            "is_top": False,
        })

    results.sort(key=lambda r: r["prob"], reverse=True)
    results[0]["is_top"] = True
    return results
