# XAI Baseline Models for 10-Class Household Object Classification

This project implements four baseline convolutional neural network models for a **10-class household object classification task**.

The purpose is to compare how different forms of pretraining (ImageNet vs. Flowers-102) and different training strategies (Linear Probing vs. Fine-Tuning) affect model performance and explainability.

---

## Project Overview

We study the effect of transfer learning under four different configurations:

| Model | Pretraining | Strategy | Description |
|-------|-------------|----------|-------------|
| **IN-LP** | ImageNet | Linear Probing | Frozen ImageNet features |
| **IN-FT** | ImageNet | Fine-Tuning | Adapted ImageNet features |
| **FL-LP** | Flowers-102 | Linear Probing | Frozen flower features |
| **FL-FT** | Flowers-102 | Fine-Tuning | Adapted flower features |

### Naming Convention

- **IN** = ImageNet pretrained
- **FL** = Flowers-102 pretrained
- **LP** = Linear Probing (only FC layer trained)
- **FT** = Fine-Tuning (all layers trained)

---

## Repository Structure

```text
block-4-5-model-ab-fatemeh/
├── notebooks/
│   └── baselinemodel_block.ipynb    # Main experiment notebook
├── data/
│   ├── ImageNetSubset/              # 10-class dataset (train/val)
│   └── xAI_ImageNet1k_OwnTestSet/   # Test set
├── models/
│   ├── best_model_A.pth             # IN-LP weights
│   ├── best_model_B.pth             # IN-FT weights
│   ├── best_model_C.pth             # FL-LP weights
│   ├── best_model_D.pth             # FL-FT weights
│   └── flower_resnet18_state.pth    # Flower pretraining checkpoint
├── scripts/
│   └── train_flower_pretrain.py     # Script to create flower-pretrained weights
├── report/
│   └── main.tex                     # LaTeX report
├── gradcam_outputs/                 # Grad-CAM visualizations
├── requirements.txt
└── README.md
```

---

## Dataset

**10 Household Object Classes:**
- binder, coffee-mug, computer-keyboard, mouse, notebook
- remote-control, soup-bowl, teapot, toilet-tissue, wooden-spoon

| Split | Images | Per Class |
|-------|--------|-----------|
| Train | 13,000 | 1,300 |
| Val | 500 | 50 |
| Test | 4,409 | ~440 |

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd block-4-5-model-ab-fatemeh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Experiments

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/baselinemodel_block.ipynb
```

### Workflow:

1. **Run Blocks 0-7** (setup and model definitions)
2. **Run Block 8** (train all models) - ~30-60 minutes
3. **Run Block 9** (test evaluation)
4. **Run Block 10** (Grad-CAM explainability)

After first training, use **Block 8.5** to load saved models instantly.

---

## Results Summary

| Model | Val Acc | Test Acc |
|-------|---------|----------|
| IN-LP | ~82% | ~41% |
| IN-FT | ~85% | ~45% |
| FL-LP | ~82% | ~41% |
| FL-FT | ~84% | ~42% |

---

## Authors

- **Fatemeh Mohammadi** - Models IN-LP & IN-FT (Blocks 4-5)

---

## License

For academic/educational use only.
