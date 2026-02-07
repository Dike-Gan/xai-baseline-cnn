# scripts/build_difficulty_table.py

from __future__ import annotations
import sys
from pathlib import Path

# add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.config import load_config, select_device, build_loaders
from src.eval.difficulty import build_difficulty_table
from models import build_model


def main():
    cfg = load_config()
    device = select_device(True)

    loaders, class_names = build_loaders(cfg)


    eval_loader = loaders["test"]

    for model_name, model_id in [
        ("Model A", "A"),
        ("Model B", "B"),
        ("Model C", "C"),
        ("Model D", "D"),
    ]:
        print(f"\nProcessing {model_name}...")

        model = build_model(model_id, len(class_names), cfg)
        ckpt_path = f"artifacts/checkpoints/best_model_{model_id}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        save_path = f"outputs/tables/difficulty_{model_id}_test_full.csv"

        build_difficulty_table(
            model=model,
            loader=eval_loader,
            class_names=class_names,
            device=device,
            save_csv=save_path,
        )

        print(f"{model_name} - saved full difficulty table to {save_path}")


if __name__ == "__main__":
    main()
