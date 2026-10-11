from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from reset_to_next_set import FEATURES
from ridge_regression import run_experiments, save_results


OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "figures"
EXPERIMENT_LABELS = {
    "set_to_reset": "SET_n to RESET_n",
    "reset_to_next_set": "RESET_n to SET_(n+1)",
}


def add_value_labels(axis, bars, values):
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_comparison(results):
    x_positions = np.arange(len(FEATURES))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    for row, (experiment_name, result) in enumerate(results.items()):
        comparison = result["test_comparison"]
        experiment_label = EXPERIMENT_LABELS[experiment_name]

        r2_axis = axes[row, 0]
        linear_r2 = comparison.loc[FEATURES, "linear_R2"].to_numpy()
        ridge_r2 = comparison.loc[FEATURES, "ridge_R2"].to_numpy()
        linear_bars = r2_axis.bar(
            x_positions - width / 2,
            linear_r2,
            width,
            label="Linear regression",
            color="#4C78A8",
        )
        ridge_bars = r2_axis.bar(
            x_positions + width / 2,
            ridge_r2,
            width,
            label=f"Ridge (alpha={result['selected_alpha']:.3g})",
            color="#F58518",
        )
        r2_axis.axhline(0, color="black", linewidth=0.8)
        r2_axis.set_title(f"{experiment_label}: test R²")
        r2_axis.set_ylabel("R²")
        r2_axis.grid(axis="y", alpha=0.25)
        r2_axis.legend(fontsize=8)
        add_value_labels(r2_axis, linear_bars, linear_r2)
        add_value_labels(r2_axis, ridge_bars, ridge_r2)

        mae_axis = axes[row, 1]
        linear_mae = comparison.loc[
            FEATURES, "linear_MAE_percent"
        ].to_numpy()
        ridge_mae = comparison.loc[
            FEATURES, "ridge_MAE_percent"
        ].to_numpy()
        linear_bars = mae_axis.bar(
            x_positions - width / 2,
            linear_mae,
            width,
            label="Linear regression",
            color="#4C78A8",
        )
        ridge_bars = mae_axis.bar(
            x_positions + width / 2,
            ridge_mae,
            width,
            label=f"Ridge (alpha={result['selected_alpha']:.3g})",
            color="#F58518",
        )
        mae_axis.set_title(f"{experiment_label}: test relative MAE")
        mae_axis.set_ylabel("Relative MAE [%]")
        mae_axis.grid(axis="y", alpha=0.25)
        add_value_labels(mae_axis, linear_bars, linear_mae)
        add_value_labels(mae_axis, ridge_bars, ridge_mae)

        for axis in axes[row]:
            axis.set_xticks(x_positions, FEATURES, rotation=18)

    figure.suptitle(
        "Linear regression and Ridge regression on chronological test cycles",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Ridge alpha is selected only on the validation set by maximizing "
            "the mean R² across the four targets."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "ridge_vs_linear_regression.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    experiment_results = run_experiments()
    save_results(experiment_results)
    generated_path = plot_comparison(experiment_results)
    print(f"Generated plot: {generated_path}")
