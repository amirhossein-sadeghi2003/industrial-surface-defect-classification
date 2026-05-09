from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


DATA_DIR = Path("data/raw/NEU-DET")
MODEL_PATH = Path("models/resnet18_best.pth")
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "gradcam_resnet18_examples.png"

IMAGE_SIZE = 224
EXAMPLES_PER_CLASS = 1


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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input_tensor, output_tensor):
        self.activations = output_tensor.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, output

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def resize_heatmap(cam, image_size):
    heatmap = Image.fromarray(np.uint8(cam * 255))
    heatmap = heatmap.resize(image_size, resample=Image.BICUBIC)
    heatmap = np.array(heatmap).astype(np.float32) / 255.0
    return heatmap


def load_display_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image).astype(np.float32) / 255.0


def create_input_tensor(image_path, transform, device):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor


def save_gradcam_grid(examples, gradcam_results, class_names):
    n_examples = len(examples)
    n_cols = 3
    n_rows = n_examples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, ((image_path, true_idx), result) in enumerate(
        zip(examples, gradcam_results)
    ):
        display_image = load_display_image(image_path)
        heatmap = resize_heatmap(result["cam"], image_size=(IMAGE_SIZE, IMAGE_SIZE))

        true_label = class_names[true_idx]
        pred_label = class_names[result["pred_idx"]]
        confidence = result["confidence"]

        axes[row_idx, 0].imshow(display_image, cmap="gray")
        axes[row_idx, 0].set_title(f"Original\nTrue: {true_label}", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(heatmap, cmap="jet")
        axes[row_idx, 1].set_title("Grad-CAM heatmap", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(display_image, cmap="gray")
        axes[row_idx, 2].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[row_idx, 2].set_title(
            f"Overlay\nPred: {pred_label} | Conf: {confidence:.2f}",
            fontsize=9,
        )
        axes[row_idx, 2].axis("off")

    fig.suptitle(
        "ResNet18 Grad-CAM Visual Explanations on NEU-DET Validation Images",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Saved Grad-CAM visualization to: {OUTPUT_PATH}")


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

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model=model, target_layer=target_layer)

    examples = collect_balanced_examples(
        dataset=val_dataset,
        class_names=class_names,
        examples_per_class=EXAMPLES_PER_CLASS,
    )

    gradcam_results = []

    for image_path, true_idx in examples:
        input_tensor = create_input_tensor(
            image_path=image_path,
            transform=transform,
            device=device,
        )

        cam, output = gradcam.generate(input_tensor=input_tensor)

        probabilities = torch.softmax(output, dim=1)
        confidence, pred_idx = probabilities.max(dim=1)

        pred_idx = pred_idx.item()
        confidence = confidence.item()

        gradcam_results.append(
            {
                "cam": cam,
                "pred_idx": pred_idx,
                "confidence": confidence,
            }
        )

        print(
            f"{image_path.name:25s} | "
            f"true={class_names[true_idx]:16s} | "
            f"pred={class_names[pred_idx]:16s} | "
            f"confidence={confidence:.4f}"
        )

    gradcam.close()

    save_gradcam_grid(
        examples=examples,
        gradcam_results=gradcam_results,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
