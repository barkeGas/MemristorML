from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from reset_to_next_set import FEATURES
from test_external_devices import run_external_test, save_results


OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "figures"
TASK_LABELS = {
    "set_to_reset": "SET_n to RESET_n",
    "reset_to_next_set": "RESET_n to SET_(n+1)",
}


def metric_matrix(result, task, metric):
    selected = result["per_device"].loc[
        result["per_device"]["scenario"].eq("matched_condition")
        & result["per_device"]["task"].eq(task)
        & result["per_device"]["model"].eq("ridge")
    ]
    matrix = selected.pivot(index="device", columns="target", values=metric)
    matrix = matrix.loc[sorted(matrix.index), FEATURES]

    pooled = result["summary"].loc[
        result["summary"]["scenario"].eq("matched_condition")
        & result["summary"]["task"].eq(task)
        & result["summary"]["model"].eq("ridge")
        & result["summary"]["aggregation"].eq("pooled")
    ].set_index("target")
    matrix.loc["Pooled"] = pooled.loc[FEATURES, metric].to_numpy()
    return matrix


def annotate_heatmap(axis, matrix, text_format):
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axis.text(
                column,
                row,
                text_format(value),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )


def plot_external_results(result):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for row, task in enumerate(["set_to_reset", "reset_to_next_set"]):
        r2_matrix = metric_matrix(result, task, "R2")
        mae_matrix = metric_matrix(result, task, "MAE_percent")

        r2_axis = axes[row, 0]
        r2_minimum = min(-1.0, float(r2_matrix.min().min()))
        r2_image = r2_axis.imshow(
            r2_matrix.to_numpy(),
            cmap="RdYlGn",
            norm=TwoSlopeNorm(vmin=r2_minimum, vcenter=0, vmax=1),
            aspect="auto",
        )
        annotate_heatmap(r2_axis, r2_matrix, lambda value: f"{value:.3f}")
        r2_axis.set_title(f"{TASK_LABELS[task]}: R²")
        figure.colorbar(r2_image, ax=r2_axis, fraction=0.046, pad=0.04)

        mae_axis = axes[row, 1]
        mae_image = mae_axis.imshow(
            mae_matrix.to_numpy(),
            cmap="YlOrRd",
            vmin=0,
            aspect="auto",
        )
        annotate_heatmap(mae_axis, mae_matrix, lambda value: f"{value:.1f}%")
        mae_axis.set_title(f"{TASK_LABELS[task]}: relative MAE")
        figure.colorbar(mae_image, ax=mae_axis, fraction=0.046, pad=0.04)

        for axis, matrix in [(r2_axis, r2_matrix), (mae_axis, mae_matrix)]:
            axis.set_xticks(np.arange(len(FEATURES)), FEATURES, rotation=18)
            axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
            axis.axhline(len(matrix.index) - 1.5, color="black", linewidth=1.4)

    alphas = result["alphas"].set_index("task")["ridge_alpha"]
    figure.suptitle(
        "Frozen Ridge models on unseen devices under the matched condition",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Training device: F2s_7_0; RESET_pw=1.0; RESET_target=100. "
            f"Alpha values remain fixed at {alphas['set_to_reset']:.3g} and "
            f"{alphas['reset_to_next_set']:.3g}."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "ridge_external_device_test.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_arguments():
    parser = ArgumentParser(
        description="Plot external-device results for the frozen Ridge models."
    )
    parser.add_argument("external_data", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    external_result = run_external_test(arguments.external_data)
    save_results(external_result)
    generated_path = plot_external_results(external_result)
    print(f"Generated plot: {generated_path}")
