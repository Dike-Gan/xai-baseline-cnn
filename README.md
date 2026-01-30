<<<<<<< HEAD
# Baseline CNN Project (ResNet-18)

This repository contains a baseline CNN project for a household object classification task.
We systematically compare four ResNet-18–based models with different pretraining strategies
and training regimes.

The project is structured for **clarity, reproducibility, and team collaboration** in a
course setting.

---

## Models Overview

We implement and compare the following four baseline models:

- **Model A**: ImageNet-pretrained ResNet-18, frozen backbone (linear probing)
- **Model B**: ImageNet-pretrained ResNet-18, full fine-tuning
- **Model C**: Flower-pretrained ResNet-18, frozen backbone (linear probing)
- **Model D**: Flower-pretrained ResNet-18, full fine-tuning

All models are trained and evaluated on the same household object dataset.

---

## Repository Structure
```
xai-baseline-cnn/
├── notebooks/
│ └── baselinemodel(block).ipynb
├── data/
│ └── ImageNetSubset/
│ ├── train/
│ └── val/
├── models/
│ └── (model checkpoints, optional)
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment Setup

We recommend using Python 3.10+ in a virtual environment.

Install dependencies via:

```bash
pip install -r requirements.txt
Dataset Preparation
The project uses an ImageNet-style folder structure and relies on
torchvision.datasets.ImageFolder.

Please place the dataset as follows:

kotlin
Copy code
data/ImageNetSubset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
Note: The dataset itself is not included in this repository.

Pretrained Flower Checkpoint (Required for Model C & D)
Models C and D require a ResNet-18 checkpoint pretrained on a flower dataset:

bash
Copy code
models/flower_resnet18_state.pth
Please download or provide this file separately and place it in the models/ directory.

Running the Experiments
All experiments are executed via the Jupyter notebook:

bash
Copy code
notebooks/baselinemodel(block).ipynb
Run the notebook cells sequentially to:

Load data

Initialize models A–D

Train each model

Evaluate validation performance

Best-performing checkpoints are automatically saved to the models/ directory.

Reproducibility
Random seed is fixed (seed = 42)

Training is performed using PyTorch

Results may vary slightly depending on hardware and CUDA configuration

Collaboration Notes
This repository intentionally uses a block-structured notebook to clearly demonstrate
individual responsibilities and model components in a team-based project.
=======
````markdown
# XAI Baseline Models for 10-Class Household Object Classification

This project implements four baseline convolutional neural network models (A, B, C, D)  
for a **10-class household object classification task**.  
The purpose is to compare how different forms of pretraining (ImageNet vs. Flower102)  
and different training strategies (linear probe vs. fine-tuning) affect model performance.

This repository is collaboratively developed by three team members using a  
branch-based Git workflow and a structured block system.

---

## 📦 Project Overview

We study the effect of transfer learning under four different configurations:

| Model | Pretraining | Backbone | Training Strategy |
|-------|-------------|----------|-------------------|
| **Model A** | ImageNet | ResNet-18 | Linear probe (backbone frozen) |
| **Model B** | ImageNet | ResNet-18 | Full fine-tuning |
| **Model C** | Flower102 | ResNet-18 | Linear probe (backbone frozen) |
| **Model D** | Flower102 | ResNet-18 | Full fine-tuning |

Models C and D use a backbone that was first fine-tuned on the  
**Oxford 102 Flowers** dataset.

The goal is to evaluate how domain-specific pretraining influences  
downstream performance on household object classification.

---

## 📂 Repository Layout

> Note: Some large datasets may **not be tracked** in Git for space reasons  
> (e.g., `102flowers/`, `data/ImageNetSubset/`).  
> They are documented here so that others can recreate the setup.

```text
xai_project/
│
├── 102flowers/                     # (optional in Git) Flower102 dataset files
├── data/
│   └── ImageNetSubset/             # (optional in Git) 10-class household subset
│
├── models/                         # Directory for saved model checkpoints (.pth)
│
├── notebooks/
│   └── baselinemodel(block).ipynb  # Collaboration notebook with block structure
│
├── flowers_dataset.py              # Custom Flower102 dataset loader
├── train_flower_pretrain.py        # Script used to create flower-pretrained weights
│
├── requirements.txt                # Clean environment dependencies
└── README.md                       # Project documentation
````

---

## 👥 Collaboration Structure (Block System)

The notebook `notebooks/baselinemodel(block).ipynb` contains **8 blocks**, each
representing a functional part of the project (dataset, training loop, models A–D, etc.).

Each team member writes their name next to the block they take responsibility for, e.g.:

```text
Block 4 – Model A  
Assigned to: Fatemeh
```

Each block is implemented in a **separate Git branch**, and then merged into `main`
via Pull Requests. This makes individual contributions and teamwork clearly visible.

---

## 🌱 Flower Pretraining

Models C and D use a backbone that was first fine-tuned on the **Oxford 102 Flowers dataset**.

The training script:

```text
train_flower_pretrain.py
```

produces a weight file such as:

```text
models/flower_resnet18_state.pth
```

During baseline training, we remove the final `fc` layer and load only the backbone weights.

---

## 🔧 Installation

Clone the repository:

```bash
git clone <repository-url>
cd xai_project
```

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows (PowerShell): .venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the project

After installing the dependencies and preparing the datasets in
`data/ImageNetSubset/` and `102flowers/`, start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/baselinemodel(block).ipynb
```

This notebook contains the **block structure** for the project.
Each team member will implement their assigned blocks in their own branch.

A separate, fully implemented training notebook can be created later
by combining the completed blocks.

---

## 🌿 Git Workflow for Team Members

### 1. Choose blocks

Open:

```text
notebooks/baselinemodel(block).ipynb
```

Add your name to your assigned blocks.

### 2. Create or switch to your feature branch

```bash
git checkout <your-branch-name>
# e.g.
# git checkout Dike
# git checkout Fatemeh
# git checkout Ruiqi
```

Make sure your branch is up to date with `main`:

```bash
git pull origin main
```

### 3. Implement your block(s)

Write code only for the blocks assigned to you.

### 4. Commit and push

```bash
git add .
git commit -m "Implement Block 4 – Model A (Fatemeh)"
git push origin <your-branch-name>
```

### 5. Open a Pull Request

Open a PR from your branch into `main`.
After review and approval, your work will be merged.

---

## ✨ Authors

* **Dike** – Project structure, flower pretraining, dataset pipeline
* **Fatemeh** – Models A & B
* **Ruiqi** – Models C & D

---

## 📄 License

For academic/exercise use only.

````
>>>>>>> c0a18d9 (Initial commit: clean CNN project skeleton)
