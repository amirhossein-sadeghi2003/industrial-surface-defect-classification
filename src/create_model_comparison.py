from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        {
            "model": "Baseline CNN",
            "best_validation_accuracy": 0.8833,
            "final_validation_accuracy": 0.87,
            "macro_f1": 0.86,
            "notes": "Custom CNN trained from scratch",
        },
        {
            "model": "ResNet18 Transfer Learning",
            "best_validation_accuracy": 0.8833,
            "final_validation_accuracy": 0.87,
            "macro_f1": 0.87,
            "notes": "Pretrained ResNet18 with frozen feature extractor",
        },
    ]

    df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    print("Model comparison:")
    print(df)
    print(f"\nSaved model comparison table to: {csv_path}")

    plot_path = RESULTS_DIR / "model_comparison.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["best_validation_accuracy"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Model Comparison on NEU Surface Defect Dataset")
    ax.tick_params(axis="x", rotation=15)

    for index, value in enumerate(df["best_validation_accuracy"]):
        ax.text(index, value + 0.015, f"{value:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved model comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()

