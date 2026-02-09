from __future__ import annotations


import torch
import sys
from pathlib import Path

# add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed, select_device, build_loaders
from models import build_model
from src.train.trainer import train_one_model

def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = select_device(True)
    loaders, class_names = build_loaders(cfg)

    num_classes = len(class_names)
    ckpt_dir = "artifacts/checkpoints"

    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")
    print(f"CKPT dir: {ckpt_dir}")

    # # ---- Model A ----
    # mA_path = f"{ckpt_dir}/best_model_A.pth"
    # if not Path(mA_path).exists():
    #     print("\n=== TRAIN Model A ===")
    #     model = build_model("A", num_classes, cfg)
    #     train_one_model(
    #         model=model,
    #         model_id="A",
    #         train_loader=loaders["train"],
    #         val_loader=loaders["val"],
    #         device=device,
    #         cfg=cfg,
    #         save_path=mA_path
    #     )

    # # ---- Model B ----
    # mB_path = f"{ckpt_dir}/best_model_B.pth"
    # if not Path(mB_path).exists():
    #     print("\n=== TRAIN Model B ===")
    #     model = build_model("B", num_classes, cfg)
    #     train_one_model(
    #         model=model,
    #         model_id="B",
    #         train_loader=loaders["train"],
    #         val_loader=loaders["val"],
    #         device=device,
    #         cfg=cfg,
    #         save_path=mB_path
    #     )

    # # ---- Model C ----
    # mC_path = f"{ckpt_dir}/best_model_C.pth"
    # if not Path(mC_path).exists():
    #     print("\n=== TRAIN Model C ===")
    #     model = build_model("C", num_classes, cfg)
    #     train_one_model(
    #         model=model,
    #         model_id="C",
    #         train_loader=loaders["train"],
    #         val_loader=loaders["val"],
    #         device=device,
    #         cfg=cfg,
    #         save_path=mC_path
    #     )

    # ---- Model D ----
    mD_path = f"{ckpt_dir}/best_model_D.pth"
    if not Path(mD_path).exists():
        print("\n=== TRAIN Model D ===")
        model = build_model("D", num_classes, cfg)
        train_one_model(
            model=model,
            model_id="D",
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            cfg=cfg,
            save_path=mD_path
        )


if __name__ == "__main__":
    main()
