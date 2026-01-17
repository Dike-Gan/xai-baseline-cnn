"""
Complete Training Script for All Models (A, B, C, D)
Run this after flower pretraining is complete
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import copy
from pathlib import Path
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============ REPRODUCIBILITY ============
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============ DEVICE SETUP ============
def select_device(prefer_cuda: bool = True) -> torch.device:
    """Select CUDA if available and usable, otherwise CPU"""
    if prefer_cuda and torch.cuda.is_available():
        try:
            _ = torch.tensor([0.0], device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"WARNING: CUDA available but not usable. Falling back to CPU.")
            print(f"   Reason: {type(e).__name__}: {e}")
            return torch.device("cpu")
    return torch.device("cpu")

device = select_device(prefer_cuda=True)
print(f"Using device: {device}")

# ============ SUPPRESS WARNINGS ============
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.backends.cudnn")

# ============ PATHS ============
FLOWER_CKPT = "../models/flower_resnet18_state.pth"
data_root = Path("data")
imagenet_root = data_root / "ImageNetSubset"
train_dir = imagenet_root / "train"
val_dir = imagenet_root / "val"

print("\nChecking data directories:")
print(f"train_dir: {train_dir.resolve()}, exists? {train_dir.exists()}")
print(f"val_dir: {val_dir.resolve()}, exists? {val_dir.exists()}")

# ============ DATA TRANSFORMS ============
input_size = 224
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# ============ LOAD DATASET ============
image_datasets = {
    "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
    "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
}
num_classes = len(image_datasets["train"].classes)
class_names = image_datasets["train"].classes

print(f"\nDataset loaded:")
print(f"Train samples: {len(image_datasets['train'])}")
print(f"Val samples: {len(image_datasets['val'])}")
print(f"Num classes: {num_classes}")
print(f"Classes: {class_names}")

# ============ CREATE DATALOADERS ============
batch_size = 32
num_workers = 0

dataloader = {
    "train": DataLoader(
        image_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ),
    "val": DataLoader(
        image_datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ),
}

# ============ TRAINING FUNCTION ============
def train_model(model, train_loader, val_loader, device, num_epochs, lr, save_path):
    """Generic training loop for a classification model"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
    )

    dataset_sizes = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
    }

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # TRAIN PHASE
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        train_epoch_loss = running_loss / dataset_sizes["train"]
        train_epoch_acc = running_corrects / dataset_sizes["train"]
        history["train_loss"].append(train_epoch_loss)
        history["train_acc"].append(train_epoch_acc)

        print(f"train  Loss: {train_epoch_loss:.4f}  Acc: {train_epoch_acc:.4f}")

        # VALIDATION PHASE
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

        val_epoch_loss = running_loss / dataset_sizes["val"]
        val_epoch_acc = running_corrects / dataset_sizes["val"]
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        print(f"val    Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

        # SAVE BEST MODEL
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            print(f"  -> New best! Saved to {save_path}")

    print(f"\nBest val Acc: {best_acc:.4f}")
    model.load_state_dict(best_state)

    return model, best_acc, history

# ============ MODEL CREATION FUNCTIONS ============
def create_model_A(num_classes):
    """Model A: Linear Probing (ImageNet pretrained, backbone frozen)"""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    print("Model A: ImageNet-pretrained backbone frozen (linear probe).")
    return model.to(device)

def create_model_B(num_classes):
    """Model B: Fine-Tuning (ImageNet pretrained, all trainable)"""
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for name, param in model.named_parameters():
        param.requires_grad = True

    print("Model B: ImageNet-pretrained backbone trainable (fine-tuning).")
    return model.to(device)

def create_model_C(num_classes):
    """Model C: Linear Probing (Flower pretrained, backbone frozen)"""
    model = models.resnet18(weights=None)

    raw_state = torch.load(FLOWER_CKPT, map_location="cpu")

    if any(k.startswith("module.") for k in raw_state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
    else:
        state = raw_state

    state_no_fc = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state_no_fc, strict=False)

    print("Loaded flower-pretrained backbone (without fc).")
    print(f"  Missing keys: {missing}")
    print(f"  Unexpected keys: {unexpected}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    print("Model C: Flower-pretrained backbone frozen (linear probe).")
    return model.to(device)

def create_model_D(num_classes):
    """Model D: Fine-Tuning (Flower pretrained, all trainable)"""
    model = models.resnet18(weights=None)

    raw_state = torch.load(FLOWER_CKPT, map_location="cpu")

    if any(k.startswith("module.") for k in raw_state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
    else:
        state = raw_state

    state_no_fc = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state_no_fc, strict=False)

    print("Loaded flower-pretrained backbone (without fc).")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for name, param in model.named_parameters():
        param.requires_grad = True

    print("Model D: Flower-pretrained backbone trainable (fine-tuning).")
    return model.to(device)

# ============ MAIN TRAINING ============
results = {}
histories = {}

print("\n" + "="*60)
print("STARTING TRAINING FOR ALL MODELS")
print("="*60)

# Check if Model A and B are already trained
model_a_path = Path("../models/best_model_A.pth")
model_b_path = Path("../models/best_model_B.pth")

if model_a_path.exists():
    print("\nModel A checkpoint found. Loading...")
    model_A = create_model_A(num_classes)
    model_A.load_state_dict(torch.load(model_a_path, map_location=device))

    # Evaluate Model A
    model_A.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_A(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    results["Model A"] = correct / total
    print(f"Model A validation accuracy: {results['Model A']:.4f}")
else:
    print("\n" + "="*60)
    print("TRAINING MODEL A")
    print("="*60)
    model_A = create_model_A(num_classes)
    model_A, best_acc_A, hist_A = train_model(
        model=model_A,
        train_loader=dataloader['train'],
        val_loader=dataloader['val'],
        device=device,
        num_epochs=10,
        lr=1e-3,
        save_path="../models/best_model_A.pth"
    )
    results["Model A"] = best_acc_A
    histories["Model A"] = hist_A

if model_b_path.exists():
    print("\nModel B checkpoint found. Loading...")
    model_B = create_model_B(num_classes)
    model_B.load_state_dict(torch.load(model_b_path, map_location=device))

    # Evaluate Model B
    model_B.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_B(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    results["Model B"] = correct / total
    print(f"Model B validation accuracy: {results['Model B']:.4f}")
else:
    print("\n" + "="*60)
    print("TRAINING MODEL B")
    print("="*60)
    model_B = create_model_B(num_classes)
    model_B, best_acc_B, hist_B = train_model(
        model=model_B,
        train_loader=dataloader['train'],
        val_loader=dataloader['val'],
        device=device,
        num_epochs=10,
        lr=1e-4,
        save_path="../models/best_model_B.pth"
    )
    results["Model B"] = best_acc_B
    histories["Model B"] = hist_B

# Check if flower checkpoint exists
flower_ckpt_path = Path(FLOWER_CKPT)
if not flower_ckpt_path.is_file():
    print(f"\nWARNING: Flower checkpoint not found at {flower_ckpt_path.resolve()}. Skipping Models C and D.")
    print("Please run train_flower_pretrain.py first to create the flower checkpoint.")
    results["Model C"] = None
    results["Model D"] = None
else:
    print("\n" + "="*60)
    print("TRAINING MODEL C")
    print("="*60)
    model_C = create_model_C(num_classes)
    model_C, best_acc_C, hist_C = train_model(
        model=model_C,
        train_loader=dataloader['train'],
        val_loader=dataloader['val'],
        device=device,
        num_epochs=10,
        lr=1e-3,
        save_path="../models/best_model_C.pth"
    )
    results["Model C"] = best_acc_C
    histories["Model C"] = hist_C

    print("\n" + "="*60)
    print("TRAINING MODEL D")
    print("="*60)
    model_D = create_model_D(num_classes)
    model_D, best_acc_D, hist_D = train_model(
        model=model_D,
        train_loader=dataloader['train'],
        val_loader=dataloader['val'],
        device=device,
        num_epochs=10,
        lr=1e-4,
        save_path="../models/best_model_D.pth"
    )
    results["Model D"] = best_acc_D
    histories["Model D"] = hist_D

# ============ RESULTS SUMMARY ============
print("\n" + "="*60)
print("TRAINING COMPLETE - FINAL RESULTS")
print("="*60)
print("\nValidation Accuracy Comparison:")
for model_name, acc in results.items():
    if acc is not None:
        print(f"{model_name}: {acc:.4f} ({acc*100:.2f}%)")
    else:
        print(f"{model_name}: Not trained (missing flower checkpoint)")

# Create results table
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Pretraining': ['ImageNet', 'ImageNet', 'Flowers-102', 'Flowers-102'],
    'Strategy': ['Linear Probe', 'Fine-Tuning', 'Linear Probe', 'Fine-Tuning'],
    'Validation Accuracy': [results[k] if results[k] is not None else 0 for k in results.keys()],
    'Accuracy (%)': [results[k]*100 if results[k] is not None else 0 for k in results.keys()]
})

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('../models/training_results.csv', index=False)
print(f"\nResults saved to: ../models/training_results.csv")

print("\n" + "="*60)
print("ALL TRAINING COMPLETE!")
print("="*60)
