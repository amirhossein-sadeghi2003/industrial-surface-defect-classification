from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


DATA_DIR = Path("data/raw/NEU-DET")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
RANDOM_SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders():
    weights = models.ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=DATA_DIR / "train" / "images",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=DATA_DIR / "validation" / "images",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


def save_training_curves(history):
    output_path = RESULTS_DIR / "resnet18_training_curves.png"

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_acc"], linestyle="--", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], linestyle="--", label="Validation Accuracy")
    ax2.set_ylabel("Accuracy")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title("ResNet18 Transfer Learning Training Curves")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved training curves to: {output_path}")


def save_confusion_matrix(y_true, y_pred, class_names):
    output_path = RESULTS_DIR / "resnet18_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=30,
        ax=ax,
        cmap="Blues",
    )
    ax.set_title("ResNet18 Transfer Learning Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved confusion matrix to: {output_path}")


def main():
    set_seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders()

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print("Using pretrained ResNet18 with frozen feature extractor.")

    model = build_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "resnet18_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, y_true, y_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    print("\nFinal validation report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    save_training_curves(history)
    save_confusion_matrix(y_true, y_pred, class_names)

    print("\nSummary:")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
