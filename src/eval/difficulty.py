# src/eval/difficulty.py

from pathlib import Path
import torch
import pandas as pd
import math
from torch.utils.data import Subset


def _get_filepath(dataset, idx):
    """
    Robustly fetch filepath from ImageFolder or Subset(ImageFolder)
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        real_idx = dataset.indices[idx]
        return base.samples[real_idx][0]

    if hasattr(dataset, "samples"):
        return dataset.samples[idx][0]

    return ""


@torch.no_grad()
def build_difficulty_table(
    model,
    loader,
    class_names,
    device,
    save_csv: str | Path,
):
    """
    Build FULL difficulty table (one row per sample in loader.dataset).
    """

    model.eval()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    rows = []
    dataset = loader.dataset
    batch_size = loader.batch_size

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)

        losses = criterion(logits, targets)
        confs, preds = probs.max(dim=1)

        # top-2 margin
        if probs.size(1) >= 2:
            top2 = torch.topk(probs, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
        else:
            margins = torch.full_like(confs, math.nan)

        prob_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        for i in range(inputs.size(0)):
            global_idx = batch_idx * batch_size + i
            filepath = _get_filepath(dataset, global_idx)

            true_idx = targets[i].item()
            pred_idx = preds[i].item()

            rows.append({
                "dataset_index": global_idx,
                "filepath": filepath,
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "true_label_idx": true_idx,
                "pred_label_idx": pred_idx,
                "is_correct": int(true_idx == pred_idx),
                "confidence": float(confs[i].item()),
                "prob_true": float(prob_true[i].item()),
                "loss": float(losses[i].item()),
                "top2_margin": float(margins[i].item()),
            })

    df = pd.DataFrame(rows)

    save_csv = Path(save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv, index=False)

    return df
