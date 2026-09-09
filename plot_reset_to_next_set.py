from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from reset_to_next_set import FEATURES, run_experiment


OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "figures"
TARGET_UNITS = {
    "start_v": "V",
    "final_v": "V",
    "initial_g": "S",
    "final_g": "S",
}


def add_value_labels(axis, bars, values):
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.3g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_predictions(result):
    y_true = result["Y_test"]
    y_predicted = result["test_predictions"]
    metrics = result["test_metrics"]

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    for axis, target in zip(axes.flatten(), FEATURES):
        actual = y_true[target]
        predicted = y_predicted[target]
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        padding = 0.05 * (upper - lower)

        axis.scatter(actual, predicted, alpha=0.4, s=20)
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            color="black",
            linewidth=1.2,
            label="Ideal: y = x",
        )
        axis.set_title(f"SET {target}")
        axis.set_xlabel(f"Actual value [{TARGET_UNITS[target]}]")
        axis.set_ylabel(
            f"Predicted value [{TARGET_UNITS[target]}]"
        )
        axis.grid(alpha=0.25)
        axis.text(
            0.04,
            0.96,
            (
                f"MSE = {metrics.loc[target, 'MSE']:.3g}\n"
                f"RMSE = {metrics.loc[target, 'RMSE']:.3g}\n"
                f"MAE% = {metrics.loc[target, 'MAE_percent']:.2f}%\n"
                f"R2 = {metrics.loc[target, 'R2']:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
        )

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Linear regression: previous-cycle RESET -> next-cycle SET",
        fontsize=14,
    )
    figure.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "linear_regression_reset_to_next_set_predictions.png"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_metrics(result):
    metrics = result["test_metrics"]
    x_positions = np.arange(len(FEATURES))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    configurations = [
        ("MSE", "Test MSE", True),
        ("RMSE", "Test RMSE", True),
        (
            "MAE_percent",
            "Test relative MAE [%]",
            False,
        ),
        ("R2", "Test R2", False),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for axis, (metric, title, logarithmic) in zip(
        axes.flatten(),
        configurations,
    ):
        values = metrics.loc[FEATURES, metric].to_numpy()
        bars = axis.bar(x_positions, values, color=colors)
        axis.set_xticks(x_positions, FEATURES, rotation=20)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

        if logarithmic:
            axis.set_yscale("log")

        if metric == "R2":
            axis.axhline(0, color="black", linewidth=0.9)

        if metric == "MAE_percent":
            axis.set_ylabel(
                "100 x MAE / mean absolute actual value"
            )

        add_value_labels(axis, bars, values)

    figure.suptitle(
        "Metrics for the RESET_n -> SET_n+1 linear model",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        (
            "MSE and RMSE use each target's units; a log scale is "
            "used because their orders of magnitude differ."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    output_path = (
        OUTPUT_DIRECTORY
        / "linear_regression_reset_to_next_set_metrics.png"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    experiment_result = run_experiment()
    prediction_path = plot_predictions(experiment_result)
    metrics_path = plot_metrics(experiment_result)
    print(f"Prediction plot: {prediction_path}")
    print(f"Metrics plot:    {metrics_path}")
