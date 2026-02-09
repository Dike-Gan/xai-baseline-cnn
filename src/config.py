# src/config.py

import yaml
import random
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(cfg):
    data_root = Path(cfg["data"]["root"])
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["input_size"], cfg["data"]["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            cfg["data"]["mean"],
            cfg["data"]["std"],
        ),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform)
    val_ds   = datasets.ImageFolder(data_root / "val", transform)
    test_ds  = datasets.ImageFolder(data_root / "test", transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return loaders, train_ds.classes
