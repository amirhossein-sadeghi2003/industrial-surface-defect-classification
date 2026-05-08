from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


DATA_DIR = Path("data/raw/NEU-DET")
RESULTS_DIR = Path("results")


def collect_image_records():
    records = []

    for split in ["train", "validation"]:
        images_dir = DATA_DIR / split / "images"

        for class_dir in sorted(images_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            for image_path in sorted(class_dir.glob("*.jpg")):
                records.append(
                    {
                        "split": split,
                        "class_name": class_name,
                        "image_path": str(image_path),
                    }
                )

    return pd.DataFrame(records)


def save_class_distribution(df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    distribution = (
        df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "class_name"])
    )

    output_csv = RESULTS_DIR / "dataset_class_distribution.csv"
    distribution.to_csv(output_csv, index=False)

    print("Class distribution:")
    print(distribution)
    print(f"\nSaved class distribution to: {output_csv}")

    return distribution


def save_distribution_plot(distribution):
    output_path = RESULTS_DIR / "dataset_class_distribution.png"

    pivot_df = distribution.pivot(index="class_name", columns="split", values="count")

    ax = pivot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("NEU Surface Defect Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved class distribution plot to: {output_path}")


def save_sample_grid(df, images_per_class=3):
    output_path = RESULTS_DIR / "sample_defect_images.png"

    class_names = sorted(df["class_name"].unique())

    fig, axes = plt.subplots(
        nrows=len(class_names),
        ncols=images_per_class,
        figsize=(images_per_class * 3, len(class_names) * 2.4),
    )

    for row_idx, class_name in enumerate(class_names):
        class_df = df[(df["split"] == "train") & (df["class_name"] == class_name)]
        sample_paths = class_df["image_path"].head(images_per_class).tolist()

        for col_idx, image_path in enumerate(sample_paths):
            image = Image.open(image_path).convert("L")
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="gray")
            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=10)

    plt.suptitle("Sample Images from NEU Surface Defect Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved sample image grid to: {output_path}")


def main():
    df = collect_image_records()

    print(f"Total images: {len(df)}")
    print()
    print("Split counts:")
    print(df["split"].value_counts())
    print()

    distribution = save_class_distribution(df)
    save_distribution_plot(distribution)
    save_sample_grid(df)


if __name__ == "__main__":
    main()
