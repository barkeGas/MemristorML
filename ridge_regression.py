from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reset_to_next_set import (
    FEATURES,
    build_temporal_pairs,
    chronological_split,
    load_data,
)


DATA_PATH = Path(__file__).with_name("F2s_7_0.parquet")
OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "results"
ALPHA_GRID = np.logspace(-6, 4, 21)


def load_same_cycle_data():
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


def same_cycle_split():
    X, Y = load_same_cycle_data()
    train_end = int(len(X) * 0.80)
    validation_end = int(len(X) * 0.90)

    return {
        "X_train": X.iloc[:train_end].copy(),
        "Y_train": Y.iloc[:train_end].copy(),
        "X_validation": X.iloc[train_end:validation_end].copy(),
        "Y_validation": Y.iloc[train_end:validation_end].copy(),
        "X_test": X.iloc[validation_end:].copy(),
        "Y_test": Y.iloc[validation_end:].copy(),
    }


def next_operation_split():
    data = load_data()
    X, Y, pair_info, _ = build_temporal_pairs(data)
    return chronological_split(X, Y, pair_info)


def predict_frame(model, X, Y):
    return pd.DataFrame(
        model.predict(X),
        index=Y.index,
        columns=FEATURES,
    )


def regression_metrics(y_true, y_predicted):
    rows = []

    for target in FEATURES:
        mse = mean_squared_error(y_true[target], y_predicted[target])
        mae = mean_absolute_error(y_true[target], y_predicted[target])
        mean_absolute_target = np.mean(np.abs(y_true[target]))
        rows.append(
            {
                "target": target,
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "MAE": mae,
                "MAE_percent": 100 * mae / mean_absolute_target,
                "R2": r2_score(y_true[target], y_predicted[target]),
            }
        )

    return pd.DataFrame(rows).set_index("target")


def select_ridge_alpha(split, alpha_grid=ALPHA_GRID):
    rows = []

    for alpha in alpha_grid:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha)),
        )
        model.fit(split["X_train"], split["Y_train"])
        predictions = predict_frame(
            model,
            split["X_validation"],
            split["Y_validation"],
        )
        target_r2 = [
            r2_score(
                split["Y_validation"][target],
                predictions[target],
            )
            for target in FEATURES
        ]
        rows.append(
            {
                "alpha": float(alpha),
                **dict(zip(FEATURES, target_r2)),
                "mean_R2": float(np.mean(target_r2)),
            }
        )

    search_results = pd.DataFrame(rows).set_index("alpha")
    selected_alpha = float(search_results["mean_R2"].idxmax())
    return selected_alpha, search_results


def fit_and_evaluate(split):
    selected_alpha, alpha_search = select_ridge_alpha(split)
    models = {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=selected_alpha),
        ),
    }
    result = {
        "selected_alpha": selected_alpha,
        "alpha_search": alpha_search,
    }

    for model_name, model in models.items():
        model.fit(split["X_train"], split["Y_train"])
        validation_predictions = predict_frame(
            model,
            split["X_validation"],
            split["Y_validation"],
        )
        test_predictions = predict_frame(
            model,
            split["X_test"],
            split["Y_test"],
        )
        result[model_name] = {
            "model": model,
            "validation_predictions": validation_predictions,
            "test_predictions": test_predictions,
            "validation_metrics": regression_metrics(
                split["Y_validation"],
                validation_predictions,
            ),
            "test_metrics": regression_metrics(
                split["Y_test"],
                test_predictions,
            ),
        }

    result["test_comparison"] = build_test_comparison(result)
    return result


def build_test_comparison(result):
    linear_metrics = result["linear"]["test_metrics"]
    ridge_metrics = result["ridge"]["test_metrics"]
    comparison = pd.DataFrame(index=FEATURES)

    for metric in ["RMSE", "MAE_percent", "R2"]:
        comparison[f"linear_{metric}"] = linear_metrics[metric]
        comparison[f"ridge_{metric}"] = ridge_metrics[metric]

    comparison["R2_change"] = (
        comparison["ridge_R2"] - comparison["linear_R2"]
    )
    comparison["RMSE_change_percent"] = 100 * (
        comparison["ridge_RMSE"] / comparison["linear_RMSE"] - 1
    )
    comparison.index.name = "target"
    return comparison


def run_experiments():
    return {
        "set_to_reset": fit_and_evaluate(same_cycle_split()),
        "reset_to_next_set": fit_and_evaluate(next_operation_split()),
    }


def print_results(results):
    for experiment_name, result in results.items():
        print(f"\n=== {experiment_name.upper()} ===")
        print(f"Selected alpha: {result['selected_alpha']:.6g}")
        print("Selection criterion: highest mean validation R2")
        print("\nTest comparison:")
        print(
            result["test_comparison"].to_string(
                float_format=lambda value: f"{value:.6g}"
            )
        )


def save_results(results):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for experiment_name, result in results.items():
        result["alpha_search"].to_csv(
            OUTPUT_DIRECTORY / f"{experiment_name}_ridge_alpha_search.csv"
        )
        result["test_comparison"].to_csv(
            OUTPUT_DIRECTORY / f"{experiment_name}_ridge_comparison.csv"
        )


if __name__ == "__main__":
    experiment_results = run_experiments()
    print_results(experiment_results)
    save_results(experiment_results)
