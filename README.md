# ResNet18 Image Classification Project

An image classification project based on ResNet18, training and evaluating 4 models with different training strategies on an ImageNet subset. The project includes complete training pipeline, model evaluation, difficulty analysis, High Confidence Wrong (HCW) analysis, and GradCAM visualization capabilities.

## Features

- **Multi-strategy Model Training**: Supports 4 different training strategies (Model A/B/C/D)
- **Model Evaluation**: Provides comprehensive test set accuracy evaluation
- **Difficulty Analysis**: Builds detailed sample difficulty tables with metrics including loss, confidence, top2 margin, etc.
- **HCW Analysis**: Identifies and analyzes High Confidence Wrong samples
- **GradCAM Visualization**: Generates model attention heatmaps to visualize model decision-making process
- **Pattern Analysis**: Analyzes failure pattern distributions across different models

## Project Structure

```
.
├── artifacts/              # Model checkpoints
│   └── checkpoints/       # Trained model weights
│       ├── best_model_A.pth
│       ├── best_model_B.pth
│       ├── best_model_C.pth
│       └── best_model_D.pth
├── configs/               # Configuration files
│   └── default.yaml       # Default configuration file
├── data/                  # Dataset directory
│   ├── ImageNetSubset/    # ImageNet subset data
│   │   ├── train/         # Training set
│   │   ├── val/           # Validation set
│   │   └── test/          # Test set
│   └── README.md          # Dataset documentation
├── models/                # Model definitions
│   ├── __init__.py
│   ├── model_factory.py   # Model factory for building Model A/B/C/D
│   └── resnet.py          # ResNet18 model implementation
├── outputs/               # Output results
│   ├── annotations/       # Annotation files (GradCAM pattern annotations)
│   ├── figures/           # Figures (pattern distribution plots, etc.)
│   ├── gradcam_teapot/    # GradCAM visualization results
│   ├── stats/             # Statistical results
│   └── tables/            # Data tables (difficulty tables, HCW distributions, etc.)
├── scripts/               # Executable scripts
│   ├── train_all.py       # Train all models
│   ├── eval.py            # Evaluate model accuracy
│   ├── build_difficulty_table.py  # Build difficulty table
│   ├── hcw_distribution_per_model.py  # HCW distribution analysis
│   ├── filter_hcw_teapot_samples_r20.py  # Filter teapot HCW samples
│   ├── run_gradcam_teapot.py  # Generate GradCAM visualizations
│   └── analyze_gradcam_patterns.py  # Analyze GradCAM patterns
├── src/                   # Core source code
│   ├── config.py          # Configuration loading and data processing
│   ├── train/             # Training related
│   │   ├── trainer.py     # Trainer
│   │   └── optim.py       # Optimizer builder
│   └── eval/              # Evaluation related
│       ├── evaluator.py   # Evaluator
│       └── difficulty.py  # Difficulty analysis
└── requirements.txt       # Python dependencies

```

## Requirements

- Python >= 3.7
- PyTorch >= 1.13
- torchvision >= 0.14
- CUDA (optional, for GPU acceleration)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd coding_part
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

**Important**: The dataset is not included in this repository and cannot be uploaded to Git due to licensing restrictions. You must prepare the ImageNet subset dataset yourself and organize it according to the project structure.

Please prepare the dataset with the following structure:

```
data/ImageNetSubset/
├── train/              # Training set images
│   ├── class1/         # One subdirectory per class
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
├── val/                # Validation set images
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/               # Test set images
    ├── class1/
    ├── class2/
    └── ...
```

**Requirements**:
- The dataset must use the `ImageFolder` format (PyTorch's default format)
- Each class should have its own subdirectory within `train/`, `val/`, and `test/` folders
- Image files can be in common formats (`.jpg`, `.jpeg`, `.png`, etc.)
- Make sure the class names are consistent across train/val/test splits

Without the properly structured dataset, the project cannot run.

### 4. Prepare Flower Pretrained Model (Required for Model C/D)

Model C and Model D require a pretrained model on the Flower dataset. 
Please download the checkpoint from the following link:
https://drive.google.com/file/d/1UsEhli2mWnZqqSzViYHte6ldmGPBS92-/view?usp=sharing
Please place the pretrained model at:

```
models/flower_resnet18_state.pth
```

## Configuration

The project uses a YAML configuration file `configs/default.yaml`. Main configuration items include:

- **seed**: Random seed (default: 42)
- **paths**: Path configurations
  - `data_root`: Dataset root directory
  - `ckpt_dir`: Checkpoint save directory
  - `flower_ckpt`: Flower pretrained model path
- **data**: Data loading configuration
  - `input_size`: Input image size (default: 224)
  - `batch_size`: Batch size (default: 32)
  - `num_workers`: Number of data loading threads
  - `mean/std`: Image normalization parameters
- **training**: Training configuration
  - `epochs`: Number of training epochs (default: 10)
  - `lr_A/B/C/D`: Learning rates for each model
  - `momentum`: SGD momentum (default: 0.9)
- **difficulty**: Difficulty analysis configuration
  - `split`: Dataset split to use ("val" or "test")
  - `top_n`: Top N sample count
- **gradcam**: GradCAM configuration
  - `split`: Dataset split to use
  - `model_name`: Model to use for generating heatmaps
  - `max_images`: Maximum number of images to generate
  - `target_layer`: Target layer name (commonly "layer4" for ResNet18)

## Usage

### 1. Train Models

Train all models (A/B/C/D):

```bash
python scripts/train_all.py
```

The script automatically checks if checkpoints exist and only trains missing models. During training, it will:
- Evaluate models on the validation set
- Save models with best validation accuracy to `artifacts/checkpoints/`
- Output training and validation metrics for each epoch

### 2. Evaluate Models

Evaluate all models on the test set:

```bash
python scripts/eval.py
```

Example output:
```
Model A: Test accuracy = 0.8523 (85.23%)
Model B: Test accuracy = 0.8765 (87.65%)
Model C: Test accuracy = 0.8234 (82.34%)
Model D: Test accuracy = 0.8456 (84.56%)
```

### 3. Build Difficulty Table

Build complete difficulty analysis tables for all models:

```bash
python scripts/build_difficulty_table.py
```

The generated CSV files contain the following fields:
- `dataset_index`: Sample index
- `filepath`: Image file path
- `true_label`: True label
- `pred_label`: Predicted label
- `true_label_idx`: True label index
- `pred_label_idx`: Predicted label index
- `is_correct`: Whether prediction is correct (0/1)
- `confidence`: Prediction confidence
- `prob_true`: Probability of true class
- `loss`: Cross-entropy loss
- `top2_margin`: Top-2 class probability margin

Output files are saved to `outputs/tables/difficulty_{model_id}_test_full.csv`

### 4. HCW Distribution Analysis

Analyze High Confidence Wrong (HCW) sample distributions for each model:

```bash
python scripts/hcw_distribution_per_model.py
```

HCW is defined as: samples with confidence > 0.9 and incorrect predictions. The script will:
- Count HCW samples per class
- Output sorted by class
- Save results to `outputs/tables/hcw_distribution_model_{model_id}.csv`

### 5. Filter Teapot HCW Samples

Filter teapot class HCW samples from all models (for GradCAM analysis):

```bash
python scripts/filter_hcw_teapot_samples_r20.py
```

The script will:
- Filter teapot class HCW samples from each model's difficulty table
- Randomly select up to 20 samples (random seed: 42)
- Save to `outputs/tables/hcw_teapot_samples/{model_id}_teapot_HCW_samples.csv`

### 6. Generate GradCAM Visualizations

Generate GradCAM heatmaps for filtered teapot HCW samples:

```bash
python scripts/run_gradcam_teapot.py
```

The script will:
- Generate visualizations for each model's teapot HCW samples
- Generate original images and GradCAM overlay images
- Save to `outputs/gradcam_teapot/model_{model_id}/`

### 7. Analyze GradCAM Patterns

Analyze GradCAM failure pattern distributions across different models:

```bash
python scripts/analyze_gradcam_patterns.py
```

**Note**: This script requires manual annotation of GradCAM image patterns beforehand. Annotation files should be placed in `outputs/annotations/gradcam_teapot/` directory, with format `{model_id}_teapot_HCW_annotated.csv`, containing a `dominant_pattern` column (values: Object/Edge/Background/Wrong-object).

The script will:
- Count pattern distributions
- Generate stacked bar chart
- Save statistical results and figures to `outputs/stats/` and `outputs/figures/`

## Model Descriptions

The project includes 4 models with different training strategies:

### Model A: ImageNet Pretrained + Linear Probing
- **Pretrained Weights**: ImageNet pretrained ResNet18
- **Training Strategy**: Freeze backbone, only train classification head (linear layer)
- **Learning Rate**: 0.001 (classification head only)
- **Characteristics**: Fast training, suitable as a baseline model

### Model B: ImageNet Pretrained + Full Fine-tuning
- **Pretrained Weights**: ImageNet pretrained ResNet18
- **Training Strategy**: Unfreeze all layers, end-to-end fine-tuning
- **Learning Rate**: 0.0001 (all layers)
- **Characteristics**: Stronger feature adaptation capability

### Model C: Flower Pretrained + Linear Probing
- **Pretrained Weights**: ResNet18 pretrained on Flower dataset
- **Training Strategy**: Freeze backbone, only train classification head
- **Learning Rate**: 0.001 (classification head only)
- **Characteristics**: Uses pretrained weights from a different domain to test transfer learning effects

### Model D: Flower Pretrained + Full Fine-tuning
- **Pretrained Weights**: ResNet18 pretrained on Flower dataset
- **Training Strategy**: Unfreeze all layers, end-to-end fine-tuning
- **Learning Rate**: 0.0001 (all layers)
- **Characteristics**: Combines advantages of cross-domain pretraining and full fine-tuning

## Output Description

### Checkpoint Files
- `artifacts/checkpoints/best_model_{A|B|C|D}.pth`: Best checkpoints for each model (based on validation accuracy)

### Difficulty Tables
- `outputs/tables/difficulty_{A|B|C|D}_test_full.csv`: Complete test set difficulty analysis tables

### HCW Analysis Results
- `outputs/tables/hcw_distribution_model_{A|B|C|D}.csv`: HCW distribution statistics for each model
- `outputs/tables/hcw_teapot_samples/{A|B|C|D}_teapot_HCW_samples.csv`: Filtered teapot HCW samples

### GradCAM Visualizations
- `outputs/gradcam_teapot/model_{A|B|C|D}/`: GradCAM visualization images for each model

### Statistical Analysis
- `outputs/stats/gradcam_teapot_pattern_summary.csv`: GradCAM pattern distribution statistics
- `outputs/figures/gradcam_teapot_pattern_distribution.png`: Pattern distribution stacked bar chart

## Technical Details

### Difficulty Metrics

- **Loss**: Cross-entropy loss, measuring model prediction uncertainty for samples
- **Confidence**: Maximum probability value of predicted class
- **Prob True**: Predicted probability of true class
- **Top2 Margin**: Difference between top-2 class probabilities, reflecting prediction certainty

### HCW (High Confidence Wrong)

High Confidence Wrong samples are those where the model makes incorrect predictions with high confidence (> 0.9). These samples typically reflect systematic biases in the model or overconfidence in certain classes.

### GradCAM

GradCAM (Gradient-weighted Class Activation Mapping) is a visualization technique that generates attention heatmaps by computing gradients of the target class with respect to feature maps, showing which image regions the model focuses on when making predictions.

## Citation

If you use this project, please cite the relevant papers and tools:

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- GradCAM: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## License

Please refer to the project license file.
