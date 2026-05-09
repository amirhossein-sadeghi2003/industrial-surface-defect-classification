from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


DATA_DIR = Path("data/raw/NEU-DET")
MODEL_PATH = Path("models/resnet18_best.pth")
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "prediction_examples_resnet18.png"

IMAGE_SIZE = 224
EXAMPLES_PER_CLASS = 2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform():
    weights = models.ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(num_classes):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    return model


def collect_balanced_examples(dataset, class_names, examples_per_class):
    selected = []

    class_counts = {class_idx: 0 for class_idx in range(len(class_names))}

    for path, class_idx in dataset.samples:
        if class_counts[class_idx] < examples_per_class:
            selected.append((Path(path), class_idx))
            class_counts[class_idx] += 1

        if all(count >= examples_per_class for count in class_counts.values()):
            break

    return selected


def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(dim=1)

    return predicted_idx.item(), confidence.item()


def save_prediction_grid(examples, predictions, class_names):
    n_examples = len(examples)
    n_cols = 4
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9))
    axes = axes.flatten()

    for ax, ((image_path, true_idx), (pred_idx, confidence)) in zip(
        axes,
        zip(examples, predictions),
    ):
        image = Image.open(image_path).convert("L")

        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        status = "Correct" if true_idx == pred_idx else "Wrong"

        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f} | {status}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[n_examples:]:
        ax.axis("off")

    fig.suptitle(
        "ResNet18 Prediction Examples on NEU-DET Validation Images",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved prediction examples to: {OUTPUT_PATH}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    transform = build_transform()

    val_dataset = datasets.ImageFolder(
        root=DATA_DIR / "validation" / "images",
        transform=transform,
    )

    class_names = val_dataset.classes
    num_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Validation images: {len(val_dataset)}")

    model = build_model(num_classes=num_classes).to(device)
    model.eval()

    examples = collect_balanced_examples(
        dataset=val_dataset,
        class_names=class_names,
        examples_per_class=EXAMPLES_PER_CLASS,
    )

    predictions = []

    for image_path, true_idx in examples:
        pred_idx, confidence = predict_image(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device,
        )
        predictions.append((pred_idx, confidence))

        print(
            f"{image_path.name:25s} | "
            f"true={class_names[true_idx]:16s} | "
            f"pred={class_names[pred_idx]:16s} | "
            f"confidence={confidence:.4f}"
        )

    save_prediction_grid(
        examples=examples,
        predictions=predictions,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
