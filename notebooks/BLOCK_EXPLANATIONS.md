# Complete Block-by-Block Explanation
## Research Question: How do different pretraining sources and training strategies affect CNN performance on household object classification?

---

## 🔷 BLOCK 0 – Imports & Setup

### WHAT does this block do?
- Loads all required Python libraries (PyTorch, torchvision, etc.)
- Sets up the computing device (CUDA GPU or CPU)
- Configures reproducibility settings (random seeds)
- Defines paths to data and model checkpoints

### WHY do we need this?
- **Reproducibility**: Setting seeds ensures experiments give the same results every time
- **Device Selection**: Use GPU if available for faster training, otherwise fallback to CPU
- **Organization**: Import everything once at the beginning to avoid errors later

### HOW does it work?
```python
# 1. Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 2. Configure PyTorch for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 3. Select device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
else:
    device = torch.device('cpu')   # Fallback to CPU

# 4. Define important paths
FLOWER_CKPT = "../models/flower_resnet18_state.pth"  # Flower pretrained weights
```

### KEY CONCEPTS:
- **`torch.manual_seed(42)`**: Makes random operations reproducible
- **`device`**: Where computations happen (GPU is ~10-100x faster than CPU)
- **cuDNN**: NVIDIA's deep learning acceleration library

---

## 🔷 BLOCK 1 – Dataset & DataLoader Setup

### WHAT does this block do?
- Loads the ImageNetSubset dataset (10 household object classes)
- Applies data augmentation to training images
- Creates DataLoaders for efficient batch processing

### WHY do we need this?
- **Data Augmentation**: Helps model generalize better by showing variations of images
  - Random flips, rotations, color changes make the model more robust
- **Normalization**: ImageNet mean/std values ensure pretrained features work properly
- **Batch Processing**: Process 32 images at once instead of 1 (much faster)

### HOW does it work?
```python
# 1. Define image transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),           # ResNet-18 requires 224×224
        transforms.RandomHorizontalFlip(),       # Flip image left-right (50% chance)
        transforms.RandomRotation(15),           # Rotate ±15 degrees randomly
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # Vary brightness/contrast
        transforms.ToTensor(),                   # Convert to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean (RGB)
                            [0.229, 0.224, 0.225]),  # ImageNet std (RGB)
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ]),
}

# 2. Load dataset
image_datasets = {
    "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
    "val": datasets.ImageFolder(val_dir, data_transforms["val"]),
}

# 3. Create DataLoaders (batch processing)
dataloader = {
    "train": DataLoader(image_datasets["train"], batch_size=32, shuffle=True),
    "val": DataLoader(image_datasets["val"], batch_size=32, shuffle=False),
}
```

### KEY CONCEPTS:
- **Data Augmentation**: Only on training data (not validation) - prevents overfitting
- **ImageFolder**: Automatically labels images based on folder structure (train/class_name/*.jpg)
- **Batch Size 32**: Process 32 images simultaneously (balance between speed and memory)
- **Shuffle**: Randomize training order each epoch (improves learning)

### DATASET STRUCTURE:
```
ImageNetSubset/
├── train/
│   ├── binder/        (1300 images)
│   ├── coffee_mug/    (1300 images)
│   └── ... (10 classes total)
└── val/
    ├── binder/        (50 images)
    ├── coffee_mug/    (50 images)
    └── ... (10 classes total)

Total: 13,000 training images, 500 validation images
```

---

## 🔷 BLOCK 2 – Training Function (train_model)

### WHAT does this block do?
- Defines a reusable training loop that works for all models (A, B, C, D)
- Handles forward pass, backward pass, optimization
- Tracks loss and accuracy for each epoch
- Saves the best model checkpoint

### WHY do we need this?
- **Code Reusability**: One function trains all 4 models (avoid copy-pasting code)
- **Automatic Checkpoint Saving**: Saves model when validation accuracy improves
- **Progress Tracking**: Monitor training/validation metrics to detect overfitting

### HOW does it work?
```python
def train_model(model, train_loader, val_loader, device, num_epochs, lr, save_path):
    """
    Generic training loop for classification

    Args:
        model: Neural network to train
        train_loader: Training data batches
        val_loader: Validation data batches
        device: 'cuda' or 'cpu'
        num_epochs: Number of training passes through dataset
        lr: Learning rate (step size for weight updates)
        save_path: Where to save best model

    Returns:
        model: Trained model (with best weights loaded)
        best_acc: Best validation accuracy achieved
        history: Dictionary of loss/accuracy per epoch
    """

    # 1. Setup optimizer and loss function
    criterion = nn.CrossEntropyLoss()  # Loss for classification
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=lr,
        momentum=0.9,  # Accelerates training in relevant direction
    )

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())  # Save initial weights

    # 2. Training loop
    for epoch in range(num_epochs):
        # === TRAINING PHASE ===
        model.train()  # Enable dropout, batch norm training mode
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()           # Reset gradients
            outputs = model(inputs)         # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            _, preds = torch.max(outputs, 1)   # Get predictions

            loss.backward()                 # Backward pass (compute gradients)
            optimizer.step()                # Update weights

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)

        # === VALIDATION PHASE ===
        model.eval()  # Disable dropout, use batch norm running stats
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():  # Don't compute gradients (saves memory)
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects / len(val_loader.dataset)

        # === SAVE BEST MODEL ===
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)

    model.load_state_dict(best_state)  # Load best weights
    return model, best_acc, history
```

### KEY CONCEPTS:
- **Forward Pass**: Input → Model → Output predictions
- **Loss Function**: Measures how wrong predictions are
- **Backward Pass**: Computes gradients (how to adjust weights)
- **Optimizer**: Updates weights using gradients
- **model.train() vs model.eval()**: Different behavior for dropout/batch norm

### WHY TWO PHASES (Train/Val)?
- **Training**: Update weights to minimize loss
- **Validation**: Check performance on unseen data (detect overfitting)
- **Never update weights on validation data!**

---

## 🔷 BLOCK 3 – Flower Pretrained Backbone Explanation

### WHAT does this block do?
- Documents how the flower checkpoint was created
- Explains the pretraining process on Oxford Flowers-102 dataset

### WHY do we need this?
- **Transparency**: Others can understand where Models C & D's weights come from
- **Reproducibility**: Documents the exact process to recreate the checkpoint
- **Research Context**: Flowers-102 is a domain-specific dataset (visual recognition of plant species)

### HOW was it created?
```python
# The flower checkpoint was created using this process:

# 1. Download Oxford Flowers-102 dataset
#    - 102 flower categories (different species)
#    - 1,020 training images
#    - 1,020 validation images
#    - 6,149 test images

# 2. Start with ImageNet pretrained ResNet-18
#    - Already knows general visual features (edges, textures, shapes)

# 3. Replace final layer
#    - Original: 1000 classes (ImageNet)
#    - New: 102 classes (Flowers)

# 4. Train on Flowers-102 for 20 epochs
#    - Learning rate: 1e-3
#    - Optimizer: SGD with momentum 0.9
#    - Scheduler: StepLR (reduce LR by 10x every 7 epochs)

# 5. Save backbone weights only (exclude final layer)
#    - Models C & D will add their own 10-class classifier
```

### KEY CONCEPTS:
- **Domain Transfer**: ImageNet (general objects) → Flowers (specific domain) → Household objects
- **Pretraining Hierarchy**:
  - ImageNet (1.2M images, 1000 classes) → broad visual features
  - Flowers-102 (8k images, 102 classes) → fine-grained visual recognition
  - Household objects (13k images, 10 classes) → target task

### WHY USE FLOWERS AS INTERMEDIATE DOMAIN?
- Tests hypothesis: Does fine-grained recognition (flower species) transfer better than general recognition (ImageNet)?
- Flowers require attention to subtle visual details (petal shape, color patterns)
- Similar to distinguishing household objects (coffee mug vs soup bowl)

---

## 🔷 BLOCK 4 – Model A (ImageNet Pretrained, Backbone Frozen)

### WHAT does this block do?
- Creates Model A for **Linear Probing** baseline
- Loads ImageNet pretrained ResNet-18
- Freezes all backbone layers
- Only trains the final classification layer

### WHY this configuration?
- **Baseline Approach**: Standard transfer learning starting point
- **Fast Training**: Only ~5,130 parameters to train (vs 11M)
- **Preserves Features**: Keeps ImageNet features intact
- **Research Question**: How well do frozen ImageNet features work?

### HOW does it work?
```python
def create_model_A(num_classes):
    # 1. Load pretrained model
    weights = ResNet18_Weights.IMAGENET1K_V1  # Official ImageNet weights
    model = models.resnet18(weights=weights)

    # 2. Replace classifier (1000 → 10 classes)
    in_features = model.fc.in_features  # 512 (ResNet-18 feature dimension)
    model.fc = nn.Linear(in_features, num_classes)  # 512 → 10

    # 3. FREEZE BACKBONE (KEY STEP!)
    for name, param in model.named_parameters():
        if not name.startswith("fc."):  # If not in final layer
            param.requires_grad = False  # Don't update these weights

    # Result:
    # - Trainable: fc.weight (512×10), fc.bias (10) = 5,130 params
    # - Frozen: All conv layers, batch norms = ~11M params

    return model
```

### MODEL A ARCHITECTURE:
```
Input Image (224×224×3)
         ↓
┌─────────────────────────────────┐
│   ResNet-18 Backbone (FROZEN)   │  ← ImageNet pretrained
│   - Conv1 (frozen)               │     Features locked
│   - Layer1 (frozen)              │
│   - Layer2 (frozen)              │
│   - Layer3 (frozen)              │
│   - Layer4 (frozen)              │
└─────────────────────────────────┘
         ↓ (512 features)
┌─────────────────────────────────┐
│   FC Layer (TRAINABLE)          │  ← Only this trains
│   Linear(512 → 10)               │
└─────────────────────────────────┘
         ↓
   Output (10 classes)
```

### TRAINING CONFIGURATION:
- **Epochs**: 10
- **Learning Rate**: 1e-3 (0.001) - Higher than fine-tuning
- **Why higher LR?**: Only training 1 layer, so can take bigger steps
- **Expected Result**: Good baseline, but may not adapt well to new domain

### KEY CONCEPTS:
- **Linear Probing**: Train only the classifier on top of frozen features
- **requires_grad = False**: Tells PyTorch "don't compute gradients for this"
- **Feature Extraction**: Backbone extracts 512-dimensional feature vectors

---

## 🔷 BLOCK 5 – Model B (ImageNet Pretrained, Full Fine-Tuning)

### WHAT does this block do?
- Creates Model B for **Fine-Tuning** comparison
- Loads ImageNet pretrained ResNet-18
- Makes ALL layers trainable (unfreezes backbone)
- Allows model to adapt ImageNet features to household objects

### WHY this configuration?
- **Maximum Adaptation**: Backbone can learn task-specific features
- **Research Question**: How much does fine-tuning improve over linear probing?
- **Expected Benefit**: Higher accuracy but slower training, risk of overfitting

### HOW does it work?
```python
def create_model_B(num_classes):
    # 1. Load pretrained model (SAME as Model A)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # 2. Replace classifier (SAME as Model A)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 3. UNFREEZE ALL LAYERS (KEY DIFFERENCE!)
    for name, param in model.named_parameters():
        param.requires_grad = True  # ALL parameters trainable

    # Result:
    # - Trainable: EVERYTHING = ~11M params
    # - Frozen: Nothing

    return model
```

### MODEL B ARCHITECTURE:
```
Input Image (224×224×3)
         ↓
┌─────────────────────────────────┐
│  ResNet-18 Backbone (TRAINABLE) │  ← ImageNet initialized
│   - Conv1 (trainable)            │     but can adapt
│   - Layer1 (trainable)           │
│   - Layer2 (trainable)           │
│   - Layer3 (trainable)           │
│   - Layer4 (trainable)           │
└─────────────────────────────────┘
         ↓ (512 features)
┌─────────────────────────────────┐
│   FC Layer (TRAINABLE)          │
│   Linear(512 → 10)               │
└─────────────────────────────────┘
         ↓
   Output (10 classes)
```

### TRAINING CONFIGURATION:
- **Epochs**: 10
- **Learning Rate**: 1e-4 (0.0001) - 10x SMALLER than Model A
- **Why lower LR?**: Avoid destroying pretrained features with large updates
- **Expected Result**: Best performance with ImageNet pretraining

### KEY COMPARISON (A vs B):
| Aspect | Model A (Linear Probe) | Model B (Fine-Tuning) |
|--------|------------------------|------------------------|
| Trainable Params | 5,130 | 11,002,890 |
| Learning Rate | 1e-3 | 1e-4 |
| Training Speed | Fast (~30 sec/epoch) | Slower (~2 min/epoch) |
| Feature Adaptation | None (frozen) | Full (adapts features) |
| Overfitting Risk | Lower | Higher |
| Expected Accuracy | Good | Better |

---

## 🔷 BLOCK 6 – Model C (Flower Pretrained, Backbone Frozen)

### WHAT does this block do?
- Creates Model C for **domain-specific pretraining** test
- Loads Flowers-102 pretrained ResNet-18 (NOT ImageNet)
- Freezes backbone (same as Model A)
- Tests if flower features transfer better than ImageNet features

### WHY this configuration?
- **Research Question**: Does domain-specific pretraining help?
- **Hypothesis**: Fine-grained recognition (flowers) may transfer better to household objects
- **Comparison**: Model C vs Model A (both linear probe, different pretraining)

### HOW does it work?
```python
def create_model_C(num_classes):
    # 1. Create EMPTY ResNet-18 (no pretrained weights yet)
    model = models.resnet18(weights=None)

    # 2. Load flower-pretrained checkpoint
    raw_state = torch.load(FLOWER_CKPT, map_location="cpu")

    # 2.1 Handle 'module.' prefix (from multi-GPU training)
    if any(k.startswith("module.") for k in raw_state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
    else:
        state = raw_state

    # 2.2 Remove fc layer weights (we'll add new classifier)
    state_no_fc = {k: v for k, v in state.items() if not k.startswith("fc.")}

    # 2.3 Load backbone only
    missing, unexpected = model.load_state_dict(state_no_fc, strict=False)
    # missing = ['fc.weight', 'fc.bias'] ← Expected (we removed these)
    # unexpected = [] ← Should be empty

    # 3. Add NEW classifier for 10 household classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # Fresh random weights

    # 4. FREEZE BACKBONE (same as Model A)
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    return model
```

### MODEL C ARCHITECTURE:
```
Input Image (224×224×3)
         ↓
┌─────────────────────────────────┐
│   ResNet-18 Backbone (FROZEN)   │  ← FLOWERS-102 pretrained
│   - Conv1 (frozen)               │     Different features than A!
│   - Layer1 (frozen)              │
│   - Layer2 (frozen)              │
│   - Layer3 (frozen)              │
│   - Layer4 (frozen)              │
└─────────────────────────────────┘
         ↓ (512 features)
┌─────────────────────────────────┐
│   FC Layer (TRAINABLE)          │  ← Only this trains
│   Linear(512 → 10)               │
└─────────────────────────────────┘
         ↓
   Output (10 classes)
```

### TRAINING CONFIGURATION:
- **Epochs**: 10
- **Learning Rate**: 1e-3 (SAME as Model A)
- **Why same LR?**: Same training strategy (linear probe)

### KEY COMPARISON (A vs C):
| Aspect | Model A (ImageNet) | Model C (Flowers) |
|--------|-------------------|-------------------|
| Pretraining Source | ImageNet-1K (1000 classes) | Flowers-102 (102 classes) |
| Pretraining Data | 1.2M images (general objects) | 8K images (flower species) |
| Feature Type | General visual features | Fine-grained recognition |
| Training Strategy | Linear Probe | Linear Probe |
| Expected Result | Good baseline | ? (research question!) |

### WHY THIS COMPARISON MATTERS:
- **Tests hypothesis**: Does domain similarity matter more than data volume?
- **Flowers-102**: Much smaller dataset but requires subtle visual discrimination
- **If Model C > Model A**: Domain-specific pretraining helps
- **If Model A > Model C**: Large-scale general pretraining is better

---

## 🔷 BLOCK 7 – Model D (Flower Pretrained, Full Fine-Tuning)

### WHAT does this block do?
- Creates Model D for **complete experimental design**
- Loads Flowers-102 pretrained ResNet-18
- Makes ALL layers trainable (same as Model B)
- Completes the 2×2 experimental matrix

### WHY this configuration?
- **Completes Design**: 2 pretraining sources × 2 training strategies = 4 models
- **Research Question**: Does fine-tuning flower features outperform all others?
- **Expected**: Best of both worlds (domain-specific + adaptation)?

### HOW does it work?
```python
def create_model_D(num_classes):
    # 1. Create EMPTY ResNet-18
    model = models.resnet18(weights=None)

    # 2. Load flower-pretrained checkpoint (SAME as Model C)
    raw_state = torch.load(FLOWER_CKPT, map_location="cpu")

    if any(k.startswith("module.") for k in raw_state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
    else:
        state = raw_state

    state_no_fc = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(state_no_fc, strict=False)

    # 3. Add NEW classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 4. UNFREEZE ALL LAYERS (KEY DIFFERENCE from Model C!)
    for name, param in model.named_parameters():
        param.requires_grad = True  # Everything trainable

    return model
```

### MODEL D ARCHITECTURE:
```
Input Image (224×224×3)
         ↓
┌─────────────────────────────────┐
│  ResNet-18 Backbone (TRAINABLE) │  ← FLOWERS-102 initialized
│   - Conv1 (trainable)            │     Can adapt to task
│   - Layer1 (trainable)           │
│   - Layer2 (trainable)           │
│   - Layer3 (trainable)           │
│   - Layer4 (trainable)           │
└─────────────────────────────────┘
         ↓ (512 features)
┌─────────────────────────────────┐
│   FC Layer (TRAINABLE)          │
│   Linear(512 → 10)               │
└─────────────────────────────────┘
         ↓
   Output (10 classes)
```

### TRAINING CONFIGURATION:
- **Epochs**: 10
- **Learning Rate**: 1e-4 (SAME as Model B)
- **Why lower LR?**: Protect flower-learned features from large updates

### COMPLETE EXPERIMENTAL DESIGN:
```
┌─────────────────┬──────────────────┬──────────────────┐
│                 │ Frozen Backbone  │ Trainable        │
│                 │ (Linear Probe)   │ (Fine-Tuning)    │
├─────────────────┼──────────────────┼──────────────────┤
│ ImageNet        │   MODEL A        │   MODEL B        │
│ Pretrained      │   LR: 1e-3       │   LR: 1e-4       │
│ (General)       │   Params: 5K     │   Params: 11M    │
├─────────────────┼──────────────────┼──────────────────┤
│ Flowers-102     │   MODEL C        │   MODEL D        │
│ Pretrained      │   LR: 1e-3       │   LR: 1e-4       │
│ (Domain-Specific│   Params: 5K     │   Params: 11M    │
└─────────────────┴──────────────────┴──────────────────┘
```

### KEY COMPARISONS:
1. **A vs B**: Effect of fine-tuning with ImageNet
2. **C vs D**: Effect of fine-tuning with Flowers
3. **A vs C**: Effect of pretraining source (linear probe)
4. **B vs D**: Effect of pretraining source (fine-tuning)
5. **Diagonal (A vs D)**: Combined effect of both factors

---

## 🔷 BLOCK 8 – Training Execution & Results

### WHAT does this block do?
- Trains all 4 models sequentially
- Saves checkpoints for each model
- Collects results into a comparison table
- Generates visualizations

### WHY this structure?
- **Systematic Execution**: Train all models with identical settings
- **Fair Comparison**: Same hardware, same data, same random seeds
- **Checkpoint Management**: Save best weights for each model

### HOW does it work?
```python
results = {}  # Store best validation accuracy for each model

# === TRAIN MODEL A ===
print("========== Training Model A ==========")
model_A = create_model_A(num_classes)
model_A, best_acc_A, hist_A = train_model(
    model=model_A,
    train_loader=dataloader['train'],
    val_loader=dataloader['val'],
    device=device,
    num_epochs=10,
    lr=1e-3,  # Higher LR for linear probe
    save_path="../models/best_model_A.pth"
)
results["Model A"] = best_acc_A

# === TRAIN MODEL B ===
print("========== Training Model B ==========")
model_B = create_model_B(num_classes)
model_B, best_acc_B, hist_B = train_model(
    model=model_B,
    train_loader=dataloader['train'],
    val_loader=dataloader['val'],
    device=device,
    num_epochs=10,
    lr=1e-4,  # Lower LR for fine-tuning
    save_path="../models/best_model_B.pth"
)
results["Model B"] = best_acc_B

# === CHECK FOR FLOWER CHECKPOINT ===
flower_ckpt_path = Path(FLOWER_CKPT)
if not flower_ckpt_path.is_file():
    print("⚠️ Flower checkpoint not found. Skipping Models C and D.")
    print("Run train_flower_pretrain.py first!")
    results["Model C"] = None
    results["Model D"] = None
else:
    # === TRAIN MODEL C ===
    print("========== Training Model C ==========")
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

    # === TRAIN MODEL D ===
    print("========== Training Model D ==========")
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

# === PRINT RESULTS ===
print("\n========== FINAL RESULTS ==========")
for model_name, acc in results.items():
    if acc is not None:
        print(f"{model_name}: {acc:.4f} ({acc*100:.2f}%)")
```

### EXECUTION ORDER:
```
1. Model A → ~5 minutes  (5K params, 10 epochs)
2. Model B → ~20 minutes (11M params, 10 epochs)
3. Model C → ~5 minutes  (5K params, 10 epochs) [if flower ckpt exists]
4. Model D → ~20 minutes (11M params, 10 epochs) [if flower ckpt exists]

Total time: ~50 minutes (with CUDA GPU)
           ~3-4 hours (with CPU only)
```

### OUTPUT FILES:
```
models/
├── best_model_A.pth  (43 MB) - Best Model A checkpoint
├── best_model_B.pth  (43 MB) - Best Model B checkpoint
├── best_model_C.pth  (43 MB) - Best Model C checkpoint
├── best_model_D.pth  (43 MB) - Best Model D checkpoint
└── flower_resnet18_state.pth (43 MB) - Flower pretrained backbone
```

---

## 🔷 BLOCK 9 – Comparison & Visualization

### WHAT does this block do?
- Creates comparison table of all results
- Generates bar charts showing performance
- Plots training curves (loss/accuracy over epochs)
- Analyzes key findings

### WHY do we need this?
- **Answer Research Question**: Which combination works best?
- **Visual Communication**: Easier to understand than numbers
- **Scientific Analysis**: Identify patterns and insights

### HOW does it work?

#### 1. RESULTS TABLE
```python
results_df = pd.DataFrame({
    'Model': ['Model A', 'Model B', 'Model C', 'Model D'],
    'Pretraining': ['ImageNet', 'ImageNet', 'Flowers-102', 'Flowers-102'],
    'Strategy': ['Linear Probe', 'Fine-Tuning', 'Linear Probe', 'Fine-Tuning'],
    'Val Accuracy': [results['Model A'], results['Model B'],
                     results['Model C'], results['Model D']],
    'Accuracy (%)': [results['Model A']*100, results['Model B']*100,
                     results['Model C']*100, results['Model D']*100]
})

print(results_df)
```

Expected output:
```
   Model  Pretraining      Strategy  Val Accuracy  Accuracy (%)
0  Model A  ImageNet    Linear Probe        0.8220         82.20
1  Model B  ImageNet    Fine-Tuning         0.8580         85.80
2  Model C  Flowers-102 Linear Probe        0.7940         79.40
3  Model D  Flowers-102 Fine-Tuning         0.8320         83.20
```

#### 2. BAR CHART
```python
plt.figure(figsize=(12, 6))
models = ['Model A\n(ImageNet LP)', 'Model B\n(ImageNet FT)',
          'Model C\n(Flowers LP)', 'Model D\n(Flowers FT)']
accuracies = [results['Model A']*100, results['Model B']*100,
              results['Model C']*100, results['Model D']*100]
colors = ['#3498DB', '#E67E22', '#27AE60', '#8E44AD']

bars = plt.bar(models, accuracies, color=colors)
plt.ylabel('Validation Accuracy (%)')
plt.title('Comparison: Pretraining Source × Training Strategy')
plt.ylim(0, 100)
plt.savefig('models_comparison.png')
```

#### 3. TRAINING CURVES
```python
plt.figure(figsize=(14, 5))

# Plot 1: Loss curves
plt.subplot(1, 2, 1)
plt.plot(hist_A['train_loss'], label='Model A')
plt.plot(hist_B['train_loss'], label='Model B')
plt.plot(hist_C['train_loss'], label='Model C')
plt.plot(hist_D['train_loss'], label='Model D')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()

# Plot 2: Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(hist_A['val_acc'], label='Model A')
plt.plot(hist_B['val_acc'], label='Model B')
plt.plot(hist_C['val_acc'], label='Model C')
plt.plot(hist_D['val_acc'], label='Model D')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()

plt.savefig('training_curves.png')
```

#### 4. KEY FINDINGS ANALYSIS
```python
# Effect of Fine-Tuning
imagenet_improvement = (results['Model B'] - results['Model A']) * 100
flower_improvement = (results['Model D'] - results['Model C']) * 100

print(f"\nEffect of Fine-Tuning:")
print(f"  ImageNet: +{imagenet_improvement:.2f}%")
print(f"  Flowers:  +{flower_improvement:.2f}%")

# Effect of Pretraining Source
lp_comparison = (results['Model C'] - results['Model A']) * 100
ft_comparison = (results['Model D'] - results['Model B']) * 100

print(f"\nEffect of Pretraining Source:")
print(f"  Linear Probe: Flowers vs ImageNet = {lp_comparison:+.2f}%")
print(f"  Fine-Tuning:  Flowers vs ImageNet = {ft_comparison:+.2f}%")
```

### EXPECTED INSIGHTS:
1. **Fine-tuning helps**: Models B and D should outperform A and C
2. **ImageNet advantage**: Larger dataset may give better general features
3. **Domain specificity**: Flowers may help or hurt depending on similarity
4. **Interaction effect**: Best combination is NOT obvious from main effects

---

## 📊 SUMMARY: Complete Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    RESEARCH PIPELINE                         │
└──────────────────────────────────────────────────────────────┘

Block 0: Setup
  └─→ Configure environment, seeds, device

Block 1: Data Loading
  └─→ Load ImageNetSubset (13K train, 500 val, 10 classes)
      Apply augmentation & normalization

Block 2: Training Function
  └─→ Define reusable train_model() function

Block 3: Documentation
  └─→ Explain flower pretraining process

Blocks 4-7: Model Creation
  ├─→ Model A: ImageNet + Linear Probe
  ├─→ Model B: ImageNet + Fine-Tuning
  ├─→ Model C: Flowers + Linear Probe
  └─→ Model D: Flowers + Fine-Tuning

Block 8: Training Execution
  └─→ Train all 4 models, save checkpoints

Block 9: Analysis & Visualization
  └─→ Compare results, generate plots, draw conclusions
```

### RESEARCH QUESTION ANSWERED:
**"How do different pretraining sources and training strategies affect CNN performance?"**

**Answer**: By comparing 4 models in a 2×2 design:
- **Main Effect 1**: Pretraining source (ImageNet vs Flowers-102)
- **Main Effect 2**: Training strategy (Linear Probe vs Fine-Tuning)
- **Interaction**: Which combination is optimal?

### SCIENTIFIC CONTRIBUTIONS:
1. **Systematic comparison** of transfer learning approaches
2. **Domain-specific pretraining** evaluation (Flowers-102)
3. **Fair experimental design** with controlled factors
4. **Reproducible methodology** with code and documentation

---

## 🎯 KEY TAKEAWAYS FOR PRESENTATION

### Block 0: "Why reproducibility matters"
- Same seeds → Same results every time
- Critical for scientific research

### Block 1: "Why data augmentation matters"
- Random flips/rotations → Better generalization
- Prevents overfitting on training set

### Block 2: "Why we need train/val split"
- Training: Learn patterns
- Validation: Check if patterns generalize
- Never update weights on validation!

### Blocks 4-5: "Linear Probe vs Fine-Tuning"
- Linear Probe: Fast, simple, good baseline
- Fine-Tuning: Slower, better results, adapts features

### Blocks 6-7: "Does pretraining source matter?"
- ImageNet: General features, large scale
- Flowers: Domain-specific, fine-grained
- Research question: Which is better for household objects?

### Block 8: "Systematic experimentation"
- Train all models identically
- Fair comparison
- Answer research question with data

### Block 9: "Interpret results scientifically"
- Not just "which is best?"
- Why? When? What patterns emerge?
- Interaction effects matter!

