# Industrial Surface Defect Classification

Computer vision project for classifying industrial surface defects using deep learning and transfer learning.

This project is planned as a visual inspection and defect classification pipeline using a real industrial surface defect image dataset.

The goal is to build an AI-focused portfolio project that connects computer vision, industrial inspection, transfer learning, and model evaluation.

---

## Project Overview

Industrial surface defects can appear in manufacturing processes such as steel production, machining, rolling, coating, and visual quality inspection.

This project will use image classification to recognize defect categories from surface images.

The planned workflow is:

```text
Industrial defect image dataset
Image preprocessing
Train / validation / test split
CNN or transfer learning model
Training curves
Confusion matrix
Sample predictions
Model interpretation
```

---

## Motivation

Visual inspection is an important part of industrial monitoring and quality control.

Traditional inspection can be slow, manual, and inconsistent. Computer vision models can help automate defect recognition from images.

This project is designed as a pure AI / computer vision project while still staying close to engineering and intelligent monitoring systems.

It complements my other portfolio projects in:

- embedded AI
- sensor-based condition monitoring
- vibration-based fault diagnosis
- IoT digital twin systems
- dynamic system modeling

---

## Planned Dataset

The intended dataset is an industrial surface defect image dataset such as the NEU Surface Defect Database.

The dataset will contain surface defect images from industrial materials such as steel.

Possible defect categories include:

- crazing
- inclusion
- patches
- pitted surface
- rolled-in scale
- scratches

Raw dataset files will be stored locally under:

```text
data/raw/
```

Raw dataset files are not tracked in Git.

---

## Planned Models

Initial model candidates:

- simple baseline CNN
- transfer learning with a pretrained vision model
- optional lightweight model comparison

Possible pretrained models:

- ResNet
- MobileNet
- EfficientNet
- ConvNeXt or other `timm` models

The first version will focus on a clean and understandable image classification pipeline.

---

## Planned Outputs

The project is expected to generate:

- training and validation accuracy curves
- training and validation loss curves
- confusion matrix
- model comparison table
- sample prediction grid
- optional Grad-CAM visualizations

These outputs will make the repository visually clear and suitable for GitHub presentation.

---

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
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Current Status

Completed:

- project folder created
- Python virtual environment created
- PyTorch and torchvision installed
- common computer vision and machine learning dependencies prepared
- local wheelhouse prepared for offline installation if needed

Next steps:

- download and organize the dataset
- inspect dataset structure
- build data loading pipeline
- train the first baseline model
- generate result plots and README visuals

---

## Dependencies

Main planned libraries:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pillow`
- `tqdm`

Optional libraries prepared for later stages:

- `opencv-python`
- `albumentations`
- `scikit-image`
- `timm`
- `torchmetrics`
- `grad-cam`

---

## How to Run

The project is still in the setup stage.

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Training and evaluation commands will be added after the dataset and model pipeline are implemented.

---

## Project Role in Portfolio

This project is intended to be a visually strong AI / computer vision project.

It adds a pure image-based deep learning project to a portfolio that already includes:

- sensor-based embedded AI
- real vibration fault diagnosis
- IoT and digital twin systems
- dynamic system simulation
- state estimation and Kalman filtering

The main purpose is to demonstrate computer vision, transfer learning, model evaluation, and visual inspection using real image data.

---

## Future Work

Planned extensions:

- add the real dataset
- implement image preprocessing
- train a baseline CNN
- train a transfer learning model
- compare model performance
- add sample prediction visualizations
- add Grad-CAM model interpretation
- improve README with dataset examples and result figures

---

## Summary

This repository is the starting point for an industrial surface defect classification project.

The final goal is to build a clean computer vision pipeline for classifying real defect images and presenting the results through clear plots, sample predictions, and model evaluation.
