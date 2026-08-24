import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
METRIC_FILES = [
    RESULTS_DIR / "baseline_cnn_metrics.json",
    RESULTS_DIR / "resnet18_metrics.json",
]
MODEL_NOTES = {
    "Baseline CNN": "Custom CNN trained from scratch",
    "ResNet18 Transfer Learning": "Pretrained backbone weights frozen",
}


def load_metrics(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run both training scripts before creating the comparison."
        )

    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def build_comparison_table():
    rows = []

    for metric_path in METRIC_FILES:
        metrics = load_metrics(metric_path)
        model_name = metrics["model"]
        rows.append(
            {
                "model": model_name,
                "best_epoch": metrics["best_epoch"],
                "best_validation_accuracy": metrics[
                    "best_validation_accuracy"
                ],
                "final_epoch_validation_accuracy": metrics[
                    "final_epoch_validation_accuracy"
                ],
                "selected_checkpoint_validation_accuracy": metrics[
                    "selected_checkpoint_validation_accuracy"
                ],
                "selected_checkpoint_macro_f1": metrics[
                    "selected_checkpoint_macro_f1"
                ],
                "notes": MODEL_NOTES[model_name],
            }
        )

    return pd.DataFrame(rows)


def save_comparison_plot(df):
    plot_path = RESULTS_DIR / "model_comparison.png"
    x_positions = np.arange(len(df))
    bar_width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    accuracy_bars = ax.bar(
        x_positions - bar_width / 2,
        df["selected_checkpoint_validation_accuracy"],
        width=bar_width,
        label="Validation accuracy",
    )
    f1_bars = ax.bar(
        x_positions + bar_width / 2,
        df["selected_checkpoint_macro_f1"],
        width=bar_width,
        label="Macro F1",
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Selected-Checkpoint Validation Performance")
    ax.set_xticks(x_positions, df["model"], rotation=12)
    ax.legend()
    ax.bar_label(accuracy_bars, fmt="%.4f", padding=3)
    ax.bar_label(f1_bars, fmt="%.4f", padding=3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved model comparison plot to: {plot_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table()
    csv_path = RESULTS_DIR / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    print("Model comparison:")
    print(comparison)
    print(f"\nSaved model comparison table to: {csv_path}")

    save_comparison_plot(comparison)


if __name__ == "__main__":
    main()
