from __future__ import annotations
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.config import load_config, select_device, build_loaders
from src.eval.evaluator import evaluate_accuracy
from models import build_model

def main():
    cfg = load_config()
    device = select_device(True)
    loaders, class_names = build_loaders(cfg)
    # ---- build test loader ----
    data_root = Path(cfg["paths"]["data_root"])
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    test_transform = transforms.Compose([
        transforms.Resize((cfg["data"]["input_size"], cfg["data"]["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            cfg["data"]["mean"],
            cfg["data"]["std"],
        ),
    ])

    test_dataset = datasets.ImageFolder(data_root / "test", test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


    num_classes = len(class_names)
    ckpt_dir = "artifacts/checkpoints"

    models_info = [
        ("Model A", "A", f"{ckpt_dir}/best_model_A.pth"),
        ("Model B", "B", f"{ckpt_dir}/best_model_B.pth"),
        ("Model C", "C", f"{ckpt_dir}/best_model_C.pth"),
        ("Model D", "D", f"{ckpt_dir}/best_model_D.pth"),
    ]

    for model_name, model_id, ckpt_path in models_info:
        if not Path(ckpt_path).exists():
            print(f"{model_name}: missing checkpoint -> skip")
            continue

        model = build_model(model_id, num_classes, cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        acc = evaluate_accuracy(model, test_loader, device)
        print(f"{model_name}: Test accuracy = {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
