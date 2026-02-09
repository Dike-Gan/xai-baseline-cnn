from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

from .optim import build_optimizer


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def train_one_model(
    model: torch.nn.Module,
    model_id: str,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    save_path: str | Path,
) -> Dict[str, Any]:
    """
    Train a single model for cfg["training"]["epochs"] epochs.
    Save best checkpoint (by val_acc) to save_path.

    Returns:
      {
        "best_val_acc": float,
        "history": {
          "train_loss": [...],
          "train_acc": [...],
          "val_loss": [...],
          "val_acc": [...],
        }
      }
    """
    model_id = model_id.upper().strip()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["training"]["epochs"])

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg, model_id)

    best_val_acc = -1.0
    best_state = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        # ================= TRAIN =================
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += x.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        # ================= VAL =================
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # ================= LOG =================
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_id}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # ================= SAVE BEST =================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f"  -> New best val_acc={best_val_acc:.4f}, saved to {save_path}")

    return {"best_val_acc": best_val_acc, "history": history}
