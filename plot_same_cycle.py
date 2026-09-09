from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("F2s_7_0.parquet")
OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "figures"
FEATURES = ["start_v", "final_v", "initial_g", "final_g"]
TARGET_UNITS = {
    "start_v": "V",
    "final_v": "V",
    "initial_g": "S",
    "final_g": "S",
}


def load_valid_data():
    data = pd.read_parquet(DATA_PATH)
    X = data.loc[:, [("SET", feature) for feature in FEATURES]].copy()
    Y = data.loc[:, [("RESET", feature) for feature in FEATURES]].copy()
    X.columns = FEATURES
    Y.columns = FEATURES

    error_flags = data.loc[
        :, [("SET", "has_error"), ("RESET", "has_error")]
    ].copy()
    error_flags.columns = ["SET_has_error", "RESET_has_error"]

    has_nan = X.isna().any(axis=1) | Y.isna().any(axis=1)
    has_error = error_flags.fillna(True).astype(bool).any(axis=1)
    valid_rows = ~(has_nan | has_error)
    return X.loc[valid_rows].copy(), Y.loc[valid_rows].copy()


def regression_results(X_train, Y_train, X_test, Y_test):
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, Y_train)
    predictions = pd.DataFrame(
        model.predict(X_test),
        index=Y_test.index,
        columns=FEATURES,
    )

    metrics = pd.DataFrame(
        {
            "MAE": [
                mean_absolute_error(Y_test[target], predictions[target])
                for target in FEATURES
            ],
            "R2": [
                r2_score(Y_test[target], predictions[target])
                for target in FEATURES
            ],
        },
        index=FEATURES,
    )
    return predictions, metrics


def plot_predictions(Y_test, predictions, metrics, split_name, output_name):
    sample_positions = np.linspace(0, len(Y_test) - 1, num=5, dtype=int)
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    for axis, target in zip(axes.flatten(), FEATURES):
        actual = Y_test[target]
        predicted = predictions[target]
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        padding = 0.05 * (upper - lower)

        axis.scatter(
            actual,
            predicted,
            alpha=0.4,
            s=20,
            label="All test cycles",
        )
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            color="black",
            linewidth=1.2,
            label="Ideal: y = x",
        )
        axis.scatter(
            actual.iloc[sample_positions],
            predicted.iloc[sample_positions],
            marker="x",
            color="#F58518",
            s=65,
            linewidths=1.8,
            label="Highlighted examples",
        )

        for position in sample_positions:
            cycle = Y_test.iloc[[position]].index.get_level_values("Cycle")[0]
            axis.annotate(
                str(cycle),
                (actual.iloc[position], predicted.iloc[position]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        axis.set_title(f"RESET {target}")
        axis.set_xlabel(f"Actual value [{TARGET_UNITS[target]}]")
        axis.set_ylabel(f"Predicted value [{TARGET_UNITS[target]}]")
        axis.grid(alpha=0.25)
        axis.text(
            0.04,
            0.96,
            (
                f"MAE = {metrics.loc[target, 'MAE']:.4g} "
                f"{TARGET_UNITS[target]}\n"
                f"R² = {metrics.loc[target, 'R2']:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
        )

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        f"Linear regression: {split_name} test set",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / output_name
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_split_comparison(chronological_metrics, random_metrics):
    x_positions = np.arange(len(FEATURES))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        x_positions - width / 2,
        chronological_metrics.loc[FEATURES, "R2"],
        width,
        label="Chronological",
    )
    axes[0].bar(
        x_positions + width / 2,
        random_metrics.loc[FEATURES, "R2"],
        width,
        label="Random",
    )
    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].set_xticks(x_positions, FEATURES, rotation=20)
    axes[0].set_ylabel("R²")
    axes[0].set_title("R²: higher is better")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    relative_mae = (
        random_metrics.loc[FEATURES, "MAE"]
        / chronological_metrics.loc[FEATURES, "MAE"]
    )
    axes[1].bar(x_positions, relative_mae)
    axes[1].axhline(
        1,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Equal error",
    )
    axes[1].set_xticks(x_positions, FEATURES, rotation=20)
    axes[1].set_ylabel("Random MAE / chronological MAE")
    axes[1].set_title("Relative MAE: lower is better")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Linear regression: effect of the data-split protocol",
        fontsize=14,
    )
    figure.tight_layout()
    output_path = (
        OUTPUT_DIRECTORY / "linear_regression_split_comparison.png"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    X, Y = load_valid_data()
    train_end = int(len(X) * 0.80)
    validation_end = int(len(X) * 0.90)

    chronological_predictions, chronological_metrics = regression_results(
        X.iloc[:train_end],
        Y.iloc[:train_end],
        X.iloc[validation_end:],
        Y.iloc[validation_end:],
    )

    X_random_train, X_random_remaining, Y_random_train, Y_random_remaining = (
        train_test_split(
            X,
            Y,
            test_size=0.20,
            random_state=42,
            shuffle=True,
        )
    )
    _, X_random_test, _, Y_random_test = train_test_split(
        X_random_remaining,
        Y_random_remaining,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )
    random_predictions, random_metrics = regression_results(
        X_random_train,
        Y_random_train,
        X_random_test,
        Y_random_test,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    generated_paths = [
        plot_predictions(
            Y.iloc[validation_end:],
            chronological_predictions,
            chronological_metrics,
            "chronological",
            "linear_regression_chronological_predictions.png",
        ),
        plot_predictions(
            Y_random_test,
            random_predictions,
            random_metrics,
            "random",
            "linear_regression_random_predictions.png",
        ),
        plot_split_comparison(chronological_metrics, random_metrics),
    ]

    for path in generated_paths:
        print(f"Generated plot: {path}")


if __name__ == "__main__":
    main()
