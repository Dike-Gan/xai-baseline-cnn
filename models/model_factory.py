from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from torchvision.models import ResNet18_Weights

from .resnet import create_resnet18, load_backbone_from_checkpoint


SUPPORTED = ["A", "B", "C", "D"]


def list_supported_models() -> List[str]:
    return SUPPORTED.copy()


def build_model(
    model_id: str,
    num_classes: int,
    cfg: Optional[dict] = None,
):
    """
    Build model A/B/C/D.

    cfg (optional):
      cfg["paths"]["flower_ckpt"] -> path to flower checkpoint for C/D
    """
    model_id = model_id.upper().strip()
    if model_id not in SUPPORTED:
        raise ValueError(f"Unknown model_id '{model_id}'. Supported: {SUPPORTED}")

    # ---- A: ImageNet pretrained + linear probe (freeze backbone)
    if model_id == "A":
        return create_resnet18(
            num_classes=num_classes,
            weights=ResNet18_Weights.IMAGENET1K_V1,
            freeze_backbone=True
        )

    # ---- B: ImageNet pretrained + finetune all
    if model_id == "B":
        return create_resnet18(
            num_classes=num_classes,
            weights=ResNet18_Weights.IMAGENET1K_V1,
            freeze_backbone=False
        )

    # ---- C/D need flower ckpt
    if cfg is None:
        raise ValueError("cfg is required for Model C/D (need flower_ckpt path).")

    flower_ckpt = Path(cfg["paths"]["flower_ckpt"])
    if not flower_ckpt.exists():
        raise FileNotFoundError(f"Flower checkpoint not found: {flower_ckpt.resolve()}")

    # base model without weights, then load backbone
    if model_id == "C":
        model = create_resnet18(num_classes=num_classes, weights=None, freeze_backbone=False)
        load_backbone_from_checkpoint(model, str(flower_ckpt), drop_fc=True)
        # freeze backbone AFTER loading
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
        return model

    if model_id == "D":
        model = create_resnet18(num_classes=num_classes, weights=None, freeze_backbone=False)
        load_backbone_from_checkpoint(model, str(flower_ckpt), drop_fc=True)
        # finetune all
        return model

    raise RuntimeError("Unreachable branch")
