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
