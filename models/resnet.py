from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def create_resnet18(
    num_classes: int,
    weights: Optional[ResNet18_Weights] = None,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a ResNet18 classifier.
    - weights: None or ResNet18_Weights.IMAGENET1K_V1
    - freeze_backbone: if True, freeze all params except fc
    """
    model = models.resnet18(weights=weights)

    # replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    return model


def load_backbone_from_checkpoint(model: nn.Module, ckpt_path: str, drop_fc: bool = True) -> None:
    """
    Load a backbone checkpoint into a ResNet18 model.
    - drop_fc: if True, ignore any 'fc.*' weights in checkpoint
    Compatible with DataParallel checkpoints (module.*).
    """
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and any(k.startswith("module.") for k in raw.keys()):
        raw = {k.replace("module.", "", 1): v for k, v in raw.items()}

    if drop_fc:
        raw = {k: v for k, v in raw.items() if not k.startswith("fc.")}

    model.load_state_dict(raw, strict=False)
