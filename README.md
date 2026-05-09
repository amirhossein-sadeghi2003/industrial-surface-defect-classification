# Industrial Surface Defect Classification

Computer vision project for **industrial surface defect classification** using PyTorch, deep learning, transfer learning, and real defect image data.

This repository is an AI-focused portfolio project for automated visual inspection in industrial environments. It demonstrates a practical workflow for classifying surface defects from image data, including dataset inspection, model training, evaluation, visualization, and model comparison.

---

## Overview

Industrial surface defects can occur during manufacturing processes such as steel production, rolling, machining, coating, and visual quality inspection.

The goal of this project is to classify surface defect images into six defect categories using deep learning models.

The current workflow is:

```text
NEU-DET surface defect dataset
→ dataset inspection
→ sample visualization
→ baseline CNN training
→ ResNet18 transfer learning
→ training curve visualization
→ confusion matrix evaluation
→ model comparison
```

---

## Motivation

Visual inspection is an important part of industrial monitoring and quality control.

Manual inspection can be slow, inconsistent, and difficult to scale. Computer vision models can help automate defect recognition by learning visual patterns from image data.

This project is designed as a pure AI / computer vision project while still staying close to engineering applications such as:

- industrial inspection
- intelligent monitoring
- quality control
- defect recognition
- AI-assisted manufacturing

It complements my other portfolio projects in embedded AI, sensor-based condition monitoring, vibration-based fault diagnosis, IoT digital twins, dynamic systems, and state estimation.

---

## Dataset

This project uses the **NEU Surface Defect Dataset (NEU-DET)**.

The dataset contains grayscale images of industrial steel surface defects.

The six defect classes are:

- `crazing`
- `inclusion`
- `patches`
- `pitted_surface`
- `rolled-in_scale`
- `scratches`

Current dataset split:

| Split | Images |
|---|---:|
| Train | 1440 |
| Validation | 360 |
| Total | 1800 |

Each class is balanced:

| Split | Images per class |
|---|---:|
| Train | 240 |
| Validation | 60 |

Raw dataset files are stored locally under:

```text
data/raw/
```

Raw dataset files are not tracked in Git.

---

## Dataset Visualization

### Sample Defect Images

![Sample Defect Images](results/sample_defect_images.png)

### Class Distribution

![Dataset Class Distribution](results/dataset_class_distribution.png)

---

## Methods

This project currently compares two deep learning approaches.

### 1. Baseline CNN

A custom convolutional neural network trained from scratch.

Main characteristics:

- grayscale input
- image size: `128 × 128`
- 3 convolution blocks
- max pooling
- dropout
- fully connected classifier
- trained directly on the NEU-DET training split

Training script:

```text
src/train_baseline_cnn.py
```

### 2. ResNet18 Transfer Learning

A pretrained ResNet18 model is used as a transfer learning baseline.

Main characteristics:

- pretrained ResNet18 backbone
- grayscale images converted to 3-channel input
- image size: `224 × 224`
- frozen feature extractor
- replaced final classification layer for 6 defect classes
- trained only the final classification layer

Training script:

```text
src/train_resnet18.py
```

---

## Results

### Model Comparison

![Model Comparison](results/model_comparison.png)

| Model | Best Validation Accuracy | Final Validation Accuracy | Macro F1-score | Notes |
|---|---:|---:|---:|---|
| Baseline CNN | 0.8833 | about 0.87 | about 0.86 | Custom CNN trained from scratch |
| ResNet18 Transfer Learning | 0.8833 | about 0.87 | about 0.87 | Frozen pretrained ResNet18 feature extractor |

Both models achieved similar validation accuracy on the current validation split.

The baseline CNN performs competitively despite being trained from scratch, while ResNet18 transfer learning provides a strong reference point for future improvements such as partial fine-tuning, stronger data augmentation, and visual explainability.

Official comparison outputs:

```text
results/model_comparison.csv
results/model_comparison.png
```

---

## Baseline CNN Results

### Training Curves

![Baseline CNN Training Curves](results/baseline_cnn_training_curves.png)

### Confusion Matrix

![Baseline CNN Confusion Matrix](results/baseline_cnn_confusion_matrix.png)

Baseline CNN validation summary:

| Metric | Value |
|---|---:|
| Best validation accuracy | 0.8833 |
| Final validation accuracy | about 0.87 |
| Macro F1-score | about 0.86 |

The baseline CNN performs strongly on classes such as `crazing`, `patches`, and `rolled-in_scale`, while `inclusion` and `pitted_surface` are more challenging.

---

## ResNet18 Transfer Learning Results

### Training Curves

![ResNet18 Training Curves](results/resnet18_training_curves.png)

### Confusion Matrix

![ResNet18 Confusion Matrix](results/resnet18_confusion_matrix.png)

ResNet18 validation summary:

| Metric | Value |
|---|---:|
| Best validation accuracy | 0.8833 |
| Final validation accuracy | about 0.87 |
| Macro F1-score | about 0.87 |

The ResNet18 model improves some class-level metrics, especially for classes such as `inclusion` and `pitted_surface`, but still shows tradeoffs between defect categories. This suggests that future work should focus on fine-tuning, augmentation, and visual interpretation.

---

## Repository Structure

```text
industrial-surface-defect-classification/
├── data/
│   └── raw/
├── models/
├── results/
│   ├── baseline_cnn_confusion_matrix.png
│   ├── baseline_cnn_training_curves.png
│   ├── dataset_class_distribution.csv
│   ├── dataset_class_distribution.png
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── resnet18_confusion_matrix.png
│   ├── resnet18_training_curves.png
│   └── sample_defect_images.png
├── src/
│   ├── create_model_comparison.py
│   ├── inspect_dataset.py
│   ├── train_baseline_cnn.py
│   └── train_resnet18.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Main Files

- `src/inspect_dataset.py`  
  Inspects the dataset, exports class distribution results, and creates sample image visualizations.

- `src/train_baseline_cnn.py`  
  Trains a custom baseline CNN and generates training curves and a confusion matrix.

- `src/train_resnet18.py`  
  Trains a ResNet18 transfer learning baseline using a frozen pretrained feature extractor.

- `src/create_model_comparison.py`  
  Creates the model comparison CSV and visualization.

- `results/sample_defect_images.png`  
  Grid of sample images from each defect class.

- `results/dataset_class_distribution.png`  
  Visualization of train and validation class counts.

- `results/baseline_cnn_training_curves.png`  
  Training and validation curves for the baseline CNN.

- `results/baseline_cnn_confusion_matrix.png`  
  Confusion matrix for the baseline CNN.

- `results/resnet18_training_curves.png`  
  Training and validation curves for ResNet18 transfer learning.

- `results/resnet18_confusion_matrix.png`  
  Confusion matrix for ResNet18 transfer learning.

- `results/model_comparison.csv`  
  Tabular comparison of trained models.

- `results/model_comparison.png`  
  Visual comparison of model-level validation performance.

---

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

If internet access is limited, dependencies can also be installed from a local wheelhouse:

```bash
pip install --no-index --find-links=/home/amir/python-wheels -r requirements.txt
```

### 3. Inspect the dataset

```bash
python src/inspect_dataset.py
```

### 4. Train the baseline CNN

```bash
python src/train_baseline_cnn.py
```

### 5. Train the ResNet18 transfer learning model

```bash
python src/train_resnet18.py
```

### 6. Create model comparison results

```bash
python src/create_model_comparison.py
```

---

## Dependencies

Main libraries:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pillow`
- `tqdm`

Additional libraries prepared for future stages:

- `opencv-python`
- `albumentations`
- `scikit-image`
- `timm`
- `torchmetrics`
- `grad-cam`

---

## Portfolio Context

This project adds a visually strong AI / computer vision component to my portfolio.

It complements projects focused on:

- embedded AI
- real sensor data
- vibration-based fault diagnosis
- IoT and digital twin systems
- dynamic system simulation
- state estimation and Kalman filtering

The broader portfolio direction is:

```text
AI / ML for Intelligent Physical Systems
```

This project shows that I can work not only with sensor and time-series data, but also with image-based deep learning and transfer learning for industrial inspection.

---

## Limitations

Current limitations:

- only classification is performed, even though annotation files are available
- raw dataset files are not included in the repository
- the current ResNet18 model uses a frozen feature extractor
- no full fine-tuning has been performed yet
- no Grad-CAM or visual explanation has been added yet
- results are based on the provided train/validation split

---

## Future Work

Planned extensions:

- add prediction example visualizations
- add Grad-CAM visual explanations
- fine-tune deeper ResNet18 layers
- compare MobileNetV2 or EfficientNet
- improve data augmentation
- use detection annotations for localization-oriented experiments
- add error analysis for difficult classes such as `inclusion`

---

## Summary

This project demonstrates industrial surface defect classification using computer vision and deep learning.

It includes dataset inspection, visual exploration, a custom baseline CNN, ResNet18 transfer learning, model evaluation, confusion matrices, and model-level comparison. The project is designed to be visually clear, practical, and suitable for GitHub portfolio presentation.
