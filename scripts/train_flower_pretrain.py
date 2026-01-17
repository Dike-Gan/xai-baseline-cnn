"""
Train ResNet-18 on Oxford Flowers-102 Dataset
This script creates a pretrained backbone for Models C and D
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy
from pathlib import Path
import warnings
import random
import numpy as np

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

# ============ DATASET SETUP ============
print("\n" + "="*60)
print("DOWNLOADING OXFORD FLOWERS-102 DATASET")
print("="*60)

data_root = Path("../data/flowers")
data_root.mkdir(parents=True, exist_ok=True)

# Image transforms
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
    "test": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Download datasets
print("Downloading training set...")
train_dataset = datasets.Flowers102(
    root=str(data_root),
    split="train",
    transform=data_transforms["train"],
    download=True
)

print("Downloading test set...")
test_dataset = datasets.Flowers102(
    root=str(data_root),
    split="test",
    transform=data_transforms["test"],
    download=True
)

print(f"\nDataset downloaded successfully!")
print(f"   Train samples: {len(train_dataset)}")
print(f"   Test samples: {len(test_dataset)}")
print(f"   Number of classes: 102")

# Create dataloaders
batch_size = 32
num_workers = 0  # Set to 0 for Windows compatibility

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# ============ MODEL SETUP ============
print("\n" + "="*60)
print("CREATING RESNET-18 MODEL")
print("="*60)

# Load ImageNet pretrained model as initialization
from torchvision.models import ResNet18_Weights
weights = ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

# Replace final layer for 102 flower classes
num_classes = 102
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)
print(f"Model created with {num_classes} output classes")

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============ TRAINING FUNCTION ============
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                num_epochs=20, device='cuda'):
    """Train the model and return best weights"""

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    dataset_sizes = {
        'train': len(train_loader.dataset),
        'test': len(test_loader.dataset)
    }

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    print("\n" + "="*60)
    print(f"TRAINING FOR {num_epochs} EPOCHS")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # ======== TRAINING PHASE ========
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        train_loss = running_loss / dataset_sizes['train']
        train_acc = running_corrects / dataset_sizes['train']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # ======== TEST PHASE ========
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

        test_loss = running_loss / dataset_sizes['test']
        test_acc = running_corrects / dataset_sizes['test']

        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best! Test Acc: {best_acc:.4f}")

        # Update learning rate
        scheduler.step()

    print("\n" + "="*60)
    print(f"TRAINING COMPLETE! Best Test Acc: {best_acc:.4f}")
    print("="*60)

    # Load best weights
    model.load_state_dict(best_state)

    return model, best_acc, history

# ============ TRAIN THE MODEL ============
num_epochs = 20  # Train for 20 epochs

model, best_acc, history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device
)

# ============ SAVE THE MODEL ============
save_dir = Path("../models")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "flower_resnet18_state.pth"

torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path.resolve()}")
print(f"   Best test accuracy: {best_acc:.4f}")

print("\n" + "="*60)
print("FLOWER PRETRAINING COMPLETE!")
print("="*60)
print("\nThis checkpoint can now be used for:")
print("  - Model C (Flower pretrained, Linear Probing)")
print("  - Model D (Flower pretrained, Fine-Tuning)")
