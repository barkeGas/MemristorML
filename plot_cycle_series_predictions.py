from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_same_cycle import load_valid_data, regression_results
from reset_to_next_set import run_experiment


OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "figures"
PLOTTED_FEATURES = ["final_g", "final_v", "start_v"]
FEATURE_UNITS = {
    "final_g": "S",
    "final_v": "V",
    "start_v": "V",
}
RESET_COLOR = "#1f77b4"
SET_COLOR = "#ff7f0e"


def run_same_cycle_experiment():
    X, Y = load_valid_data()
    train_end = int(len(X) * 0.80)
    test_start = int(len(X) * 0.90)

    predictions, _ = regression_results(
        X.iloc[:train_end],
        Y.iloc[:train_end],
        X.iloc[test_start:],
        Y.iloc[test_start:],
    )

    return {
        "set_actual": X.iloc[test_start:].copy(),
        "reset_actual": Y.iloc[test_start:].copy(),
        "reset_predicted": predictions,
    }


def build_aligned_test_series():
    same_cycle = run_same_cycle_experiment()
    temporal = run_experiment()

    common_index = same_cycle["reset_actual"].index.intersection(
        temporal["Y_test"].index,
        sort=False,
    )
    if common_index.empty:
        raise ValueError("The two test sets have no cycles in common.")

    series = {
        "reset_actual": same_cycle["reset_actual"].loc[common_index],
        "reset_predicted": same_cycle["reset_predicted"].loc[common_index],
        "set_actual": temporal["Y_test"].loc[common_index],
        "set_predicted": temporal["test_predictions"].loc[common_index],
    }

    reference_index = series["reset_actual"].index
    if not all(frame.index.equals(reference_index) for frame in series.values()):
        raise ValueError("Actual and predicted series are not index-aligned.")

    cycles = reference_index.get_level_values("Cycle").to_numpy()
    if not np.all(np.diff(cycles) == 1):
        raise ValueError("The aligned test cycles are not consecutive.")

    return cycles, series


def configure_axes(axes):
    for axis, feature in zip(axes, PLOTTED_FEATURES):
        axis.set_title(feature)
        axis.grid(alpha=0.25)

        if feature == "final_g":
            axis.set_yscale("log")
            axis.set_ylabel("Conductance [S]\n(log scale)")
        else:
            axis.set_ylabel("Voltage [V]")

    axes[-1].set_xlabel("Cycle")


def finish_figure(figure, axes, title, note, output_name):
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    figure.suptitle(title, fontsize=15, y=0.985)
    figure.text(0.5, 0.012, note, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.91))

    output_path = OUTPUT_DIRECTORY / output_name
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_cycle_series(cycles, series):
    figure, axes = plt.subplots(
        len(PLOTTED_FEATURES),
        1,
        figsize=(14, 10),
        sharex=True,
    )

    line_settings = [
        (
            "reset_actual",
            "Measured RESET",
            RESET_COLOR,
            "solid",
            1.35,
            0.82,
        ),
        (
            "reset_predicted",
            "Predicted RESET",
            RESET_COLOR,
            "solid",
            1.25,
            1.0,
        ),
        (
            "set_actual",
            "Measured SET",
            SET_COLOR,
            "solid",
            1.35,
            0.82,
        ),
        (
            "set_predicted",
            "Predicted SET",
            SET_COLOR,
            "solid",
            1.25,
            1.0,
        ),
    ]

    for axis, feature in zip(axes, PLOTTED_FEATURES):
        for key, label, color, linestyle, linewidth, alpha in line_settings:
            axis.plot(
                cycles,
                series[key][feature].to_numpy(),
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

    configure_axes(axes)
    return finish_figure(
        figure,
        axes,
        "Measured and predicted switching metrics over chronological test cycles",
        (
            "Solid lines are measured values; dashed lines are linear-regression "
            "predictions. Only held-out test cycles are shown."
        ),
        "linear_regression_cycle_series_predictions.png",
    )


if __name__ == "__main__":
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    test_cycles, test_series = build_aligned_test_series()
    generated_path = plot_cycle_series(test_cycles, test_series)
    print(
        f"Plotted {len(test_cycles)} chronological test cycles: "
        f"{test_cycles[0]}-{test_cycles[-1]}"
    )
    print(f"Generated plot: {generated_path}")
