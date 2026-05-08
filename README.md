# Industrial Surface Defect Classification

Computer vision project for classifying industrial surface defects using deep learning and transfer learning.

This repository is being developed as an AI-focused portfolio project for automated visual inspection in industrial environments. The project uses a real surface defect image dataset and builds a defect classification pipeline with PyTorch.

## Project Goal

The goal of this project is to build an end-to-end image classification workflow for industrial defect inspection by combining:

- computer vision
- image classification
- deep learning
- transfer learning
- industrial AI
- model evaluation and visualization

This project is intended to be one of the more visually impressive repositories in my portfolio.

## Dataset

This project currently uses the **NEU Surface Defect Dataset (NEU-DET)**.

The dataset contains **6 defect classes**:

- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Current dataset split used in the project:

- **train:** 1440 images
- **validation:** 360 images
- **total:** 1800 images

Each class is balanced:

- 240 training images per class
- 60 validation images per class

## Dataset Visualization

### Sample Defect Images

![Sample Defect Images](results/sample_defect_images.png)

### Class Distribution

![Dataset Class Distribution](results/dataset_class_distribution.png)

## Current Project Status

The project currently includes:

- dataset download and local organization
- dataset inspection script
- class distribution export
- sample image visualization
- baseline CNN training pipeline
- training curve visualization
- confusion matrix visualization

## Baseline CNN

A first baseline convolutional neural network has been trained on the NEU-DET dataset.

### Baseline Validation Performance

- **Best validation accuracy:** `0.8833`
- **Final validation accuracy:** about `0.87`

### Classification Report

- crazing: strong performance
- patches: very strong performance
- rolled-in_scale: strong recall
- inclusion: currently more difficult
- pitted_surface: moderate performance
- scratches: good overall performance

This baseline provides a solid starting point before moving to transfer learning models such as ResNet18.

## Training Results

### Training Curves

![Baseline CNN Training Curves](results/baseline_cnn_training_curves.png)

### Confusion Matrix

![Baseline CNN Confusion Matrix](results/baseline_cnn_confusion_matrix.png)

## Repository Structure

```text
industrial-surface-defect-classification/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── results/
├── src/
│   ├── inspect_dataset.py
│   └── train_baseline_cnn.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Main Files

Important files currently included in the project:

- `src/inspect_dataset.py`
- `src/train_baseline_cnn.py`
- `results/sample_defect_images.png`
- `results/dataset_class_distribution.csv`
- `results/dataset_class_distribution.png`
- `results/baseline_cnn_training_curves.png`
- `results/baseline_cnn_confusion_matrix.png`

## How to Run

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Inspect the dataset

```bash
python src/inspect_dataset.py
```

### 4. Train the baseline CNN

```bash
python src/train_baseline_cnn.py
```

## Current Results Summary

At this stage, the project demonstrates that:

- a real industrial defect dataset has been integrated successfully
- the dataset structure has been inspected and visualized
- a baseline deep learning model can achieve promising performance
- the project already has meaningful visual outputs for portfolio presentation

## Next Steps

Planned next improvements:

- add **transfer learning** with ResNet18
- compare baseline CNN vs pretrained model
- improve augmentation strategy
- save prediction examples
- add Grad-CAM visual explanations
- improve error analysis for hard classes like `inclusion`

## Portfolio Relevance

This project strengthens my portfolio in:

- artificial intelligence
- computer vision
- industrial inspection
- deep learning with PyTorch
- practical ML experimentation
- intelligent physical systems
