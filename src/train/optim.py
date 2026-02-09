from __future__ import annotations

import torch
import torch.optim as optim


def build_optimizer(model: torch.nn.Module, cfg: dict, model_id: str) -> torch.optim.Optimizer:
    """
    Build optimizer for a given model_id (A/B/C/D).
    Default: SGD + momentum.

    Expect cfg structure like:
      cfg["training"]["momentum"]
      cfg["training"]["lr_A"], lr_B, lr_C, lr_D
    """
    model_id = model_id.upper().strip()
    lr_key = f"lr_{model_id}"
    if lr_key not in cfg["training"]:
        raise KeyError(f"Missing cfg['training']['{lr_key}']")

    lr = float(cfg["training"][lr_key])
    momentum = float(cfg["training"].get("momentum", 0.9))

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    return optimizer
