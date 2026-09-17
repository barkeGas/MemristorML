from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_same_cycle import FEATURES, load_valid_data
from reset_to_next_set import run_experiment


OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "results"
PARAMETER_ROWS = ["bias", *FEATURES]


def fit_same_cycle_model():
    X, Y = load_valid_data()
    train_end = int(len(X) * 0.80)
    test_start = int(len(X) * 0.90)

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X.iloc[:train_end], Y.iloc[:train_end])
    return model, X.iloc[test_start:].copy()


def extract_parameter_matrices(model):
    scaler = model.named_steps["standardscaler"]
    regression = model.named_steps["linearregression"]

    standardized_weights = regression.coef_
    standardized_bias = regression.intercept_
    standardized_matrix = np.vstack(
        [standardized_bias, standardized_weights.T]
    )

    original_weights = standardized_weights / scaler.scale_[np.newaxis, :]
    original_bias = standardized_bias - original_weights @ scaler.mean_
    original_matrix = np.vstack([original_bias, original_weights.T])

    matrices = {
        "standardized": pd.DataFrame(
            standardized_matrix,
            index=PARAMETER_ROWS,
            columns=FEATURES,
        ),
        "original": pd.DataFrame(
            original_matrix,
            index=PARAMETER_ROWS,
            columns=FEATURES,
        ),
    }
    for matrix in matrices.values():
        matrix.index.name = "parameter"
    return matrices


def verify_parameter_matrix(model, X, original_matrix):
    augmented_X = np.column_stack(
        [np.ones(len(X)), X.loc[:, FEATURES].to_numpy()]
    )
    manual_predictions = augmented_X @ original_matrix.to_numpy()
    sklearn_predictions = model.predict(X.loc[:, FEATURES])
    return float(np.max(np.abs(manual_predictions - sklearn_predictions)))


def print_matrix(title, matrix):
    print(f"\n=== {title} ===")
    print(matrix.to_string(float_format=lambda value: f"{value:.8g}"))


def save_matrix(matrix, filename):
    output_path = OUTPUT_DIRECTORY / filename
    matrix.to_csv(output_path, float_format="%.17g")
    return output_path


def main():
    same_cycle_model, same_cycle_test_X = fit_same_cycle_model()
    temporal_result = run_experiment()

    experiments = {
        "set_to_reset": {
            "model": same_cycle_model,
            "test_X": same_cycle_test_X,
        },
        "reset_to_next_set": {
            "model": temporal_result["model"],
            "test_X": temporal_result["X_test"],
        },
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for experiment_name, experiment in experiments.items():
        matrices = extract_parameter_matrices(experiment["model"])
        maximum_difference = verify_parameter_matrix(
            experiment["model"],
            experiment["test_X"],
            matrices["original"],
        )

        print(f"\n######## {experiment_name} ########")
        print_matrix(
            "B_z: STANDARDIZED-INPUT AUGMENTED MATRIX",
            matrices["standardized"],
        )
        print_matrix(
            "B_x: ORIGINAL-INPUT AUGMENTED MATRIX",
            matrices["original"],
        )
        print(
            "Maximum |manual matrix prediction - scikit-learn prediction|: "
            f"{maximum_difference:.3e}"
        )

        for matrix_name, matrix in matrices.items():
            output_path = save_matrix(
                matrix,
                f"{experiment_name}_parameters_{matrix_name}.csv",
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
