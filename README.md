# Baseline CNN Project (ResNet-18)

This repository contains a baseline CNN project for a household object classification task.
We systematically compare four ResNet-18–based models with different pretraining strategies
and training regimes.

The project is structured to emphasize clarity, reproducibility, and transparent team collaboration
in an academic course setting.

---

## Models Overview

We implement and compare the following four baseline models:

- Model A: ImageNet-pretrained ResNet-18, frozen backbone (linear probing)
- Model B: ImageNet-pretrained ResNet-18, full fine-tuning
- Model C: Flower-pretrained ResNet-18, frozen backbone (linear probing)
- Model D: Flower-pretrained ResNet-18, full fine-tuning

All models are trained and evaluated on the same household object dataset.

---

## Repository Structure

xai-baseline-cnn/
├── notebooks/
│   └── baselinemodel_block.ipynb
├── data/
│   └── ImageNetSubset/
│       ├── train/
│       └── val/
├── models/
│   └── (model checkpoints, optional)
├── requirements.txt
├── .gitignore
└── README.md

---

## Project Structure and Collaboration

This project follows a block-based collaboration strategy to ensure clear task separation,
reproducibility, and clean version control.

### Branching Strategy

main  
Contains a verified reference baseline and shared infrastructure.  
Should not be modified directly for model-specific development.

block-0-3-core-dike  
Responsible for Blocks 0–3 (shared configuration, data handling, training utilities).  
Maintained by Dike.

block-4-5-model-ab-fatemeh  
Responsible for Blocks 4–5 (Model A and Model B).  
Maintained by Fatemeh.

block-6-7-model-cd-ruiqi  
Responsible for Blocks 6–7 (Model C and Model D).  
Maintained by Ruiqi.

(Optional) block-8-train-eval-dike  
Used for experiment orchestration and evaluation logic.

### Collaboration Rules

Shared blocks must not be modified in individual model branches.  
Changes to shared components are discussed and integrated via the main branch.  
Final integration is done by merging completed block branches back into main.

---

## Environment Setup

We recommend using Python 3.10 or newer in a virtual environment.

Install dependencies using:

pip install -r requirements.txt

---

## Dataset Preparation

The project uses an ImageNet-style folder structure and relies on torchvision.datasets.ImageFolder.

Place the dataset as follows:

data/ImageNetSubset/
├── train/
│   ├── class_1/
│   └── ...
└── val/
    ├── class_1/
    └── ...

The dataset itself is not included in this repository.

---

## Pretrained Flower Checkpoint

Models C and D require a ResNet-18 checkpoint pretrained on a flower dataset.

Place the file here:

models/flower_resnet18_state.pth

---

## Running the Experiments

All experiments are executed via the Jupyter notebook:

notebooks/baselinemodel_block.ipynb

Run the notebook cells sequentially to train and evaluate all models.

---

## Reproducibility

Random seed is fixed (seed = 42).  
Training is performed using PyTorch.  
Results may vary slightly depending on hardware and CUDA configuration.

---

## Collaboration Notes

This repository uses a block-structured notebook to clearly demonstrate individual responsibilities
and model components in a team-based project.
