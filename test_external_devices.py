from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from reset_to_next_set import FEATURES, build_temporal_pairs
from ridge_regression import (
    next_operation_split,
    regression_metrics,
    run_experiments,
    same_cycle_split,
)


SOURCE_DEVICE = "F2s_7_0"
SOURCE_RESET_PW = 1.0
SOURCE_RESET_TARGET = 100
GROUP_LEVELS = ["RESET_pw", "RESET_target", "Device"]
EXPECTED_INDEX_LEVELS = [*GROUP_LEVELS, "Cycle"]
OUTPUT_DIRECTORY = Path(__file__).with_name("docs") / "results"


def validate_schema(data):
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("The external dataset must have a MultiIndex.")
    if data.index.names != EXPECTED_INDEX_LEVELS:
        raise ValueError(
            "Unexpected index levels: "
            f"{data.index.names}; expected {EXPECTED_INDEX_LEVELS}."
        )

    required_columns = {
        (operation, feature)
        for operation in ["SET", "RESET"]
        for feature in [*FEATURES, "has_error"]
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"The external dataset is missing columns: {sorted(missing_columns)}"
        )
    if not data.index.is_unique:
        raise ValueError("The external dataset contains duplicate indices.")

    numeric_columns = [
        (operation, feature)
        for operation in ["SET", "RESET"]
        for feature in FEATURES
    ]
    numeric_values = data.loc[:, numeric_columns].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("The external dataset contains infinite feature values.")


def build_same_cycle_pairs(data):
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
    diagnostics = {
        "input_rows": len(data),
        "rows_with_nan": int(has_nan.sum()),
        "rows_with_error": int(has_error.sum()),
        "valid_pairs": int(valid_rows.sum()),
    }
    return (
        X.loc[valid_rows].copy(),
        Y.loc[valid_rows].copy(),
        diagnostics,
    )


def build_following_operation_pairs(data):
    X, Y, _, diagnostics = build_temporal_pairs(data)
    renamed_diagnostics = {
        "input_rows": diagnostics["rows"],
        "rows_without_previous_operation": diagnostics[
            "without_previous_reset"
        ],
        "nonconsecutive_pairs": diagnostics["nonconsecutive"],
        "candidate_pairs_with_nan": diagnostics["with_nan"],
        "candidate_pairs_with_error": diagnostics["with_error"],
        "valid_pairs": diagnostics["valid_pairs"],
    }
    return X, Y, renamed_diagnostics


def select_unseen_devices(data):
    devices = data.index.get_level_values("Device")
    unseen = data.loc[devices != SOURCE_DEVICE].copy()

    pw = unseen.index.get_level_values("RESET_pw")
    target = unseen.index.get_level_values("RESET_target")
    matched_condition = unseen.loc[
        (pw == SOURCE_RESET_PW) & (target == SOURCE_RESET_TARGET)
    ].copy()

    if matched_condition.empty:
        raise ValueError(
            "No unseen devices use the source experimental condition."
        )

    return {
        "matched_condition": matched_condition,
        "all_unseen_devices": unseen,
    }


def prediction_frame(model, X, Y):
    return pd.DataFrame(
        model.predict(X),
        index=Y.index,
        columns=FEATURES,
    )


def metric_rows(metrics, common_fields):
    rows = []
    for target in FEATURES:
        rows.append(
            {
                **common_fields,
                "target": target,
                **metrics.loc[target].to_dict(),
            }
        )
    return rows


def evaluate_models(models, X, Y, scenario_name, task_name):
    summary_rows = []
    device_rows = []
    devices = X.index.get_level_values("Device")
    unique_devices = devices.unique()

    for model_name, model in models.items():
        predictions = prediction_frame(model, X, Y)
        pooled_metrics = regression_metrics(Y, predictions)
        summary_rows.extend(
            metric_rows(
                pooled_metrics,
                {
                    "scenario": scenario_name,
                    "task": task_name,
                    "model": model_name,
                    "aggregation": "pooled",
                    "rows": len(X),
                    "devices": len(unique_devices),
                },
            )
        )

        current_device_rows = []
        for device in unique_devices:
            device_mask = devices == device
            device_metrics = regression_metrics(
                Y.loc[device_mask],
                predictions.loc[device_mask],
            )
            rows = metric_rows(
                device_metrics,
                {
                    "scenario": scenario_name,
                    "task": task_name,
                    "model": model_name,
                    "device": device,
                    "rows": int(device_mask.sum()),
                },
            )
            current_device_rows.extend(rows)
            device_rows.extend(rows)

        current_devices = pd.DataFrame(current_device_rows)
        metric_columns = ["MSE", "RMSE", "MAE", "MAE_percent", "R2"]
        macro_metrics = current_devices.groupby("target")[
            metric_columns
        ].mean()
        summary_rows.extend(
            metric_rows(
                macro_metrics,
                {
                    "scenario": scenario_name,
                    "task": task_name,
                    "model": model_name,
                    "aggregation": "device_macro_mean",
                    "rows": len(X),
                    "devices": len(unique_devices),
                },
            )
        )

    return summary_rows, device_rows


def distribution_shift_rows(X_train, X_external, scenario_name, task_name):
    rows = []
    train_minimum = X_train.loc[:, FEATURES].min()
    train_maximum = X_train.loc[:, FEATURES].max()

    for feature in FEATURES:
        outside = X_external[feature].lt(train_minimum[feature]) | X_external[
            feature
        ].gt(train_maximum[feature])
        rows.append(
            {
                "scenario": scenario_name,
                "task": task_name,
                "feature": feature,
                "training_minimum": train_minimum[feature],
                "training_maximum": train_maximum[feature],
                "external_minimum": X_external[feature].min(),
                "external_maximum": X_external[feature].max(),
                "outside_training_range_percent": 100 * outside.mean(),
            }
        )

    return rows


def source_copy_matches(data, source_data):
    source_mask = data.index.get_level_values("Device") == SOURCE_DEVICE
    source_copy = data.loc[source_mask].sort_index().sort_index(axis=1)
    source_reference = source_data.sort_index().sort_index(axis=1)
    return source_copy.equals(source_reference)


def run_external_test(external_path):
    external_data = pd.read_parquet(external_path)
    validate_schema(external_data)

    fitted_results = run_experiments()
    models = {
        task: {
            model_name: fitted_results[task][model_name]["model"]
            for model_name in ["linear", "ridge"]
        }
        for task in ["set_to_reset", "reset_to_next_set"]
    }
    training_splits = {
        "set_to_reset": same_cycle_split(),
        "reset_to_next_set": next_operation_split(),
    }
    scenarios = select_unseen_devices(external_data)

    summary_rows = []
    device_rows = []
    diagnostic_rows = []
    shift_rows = []

    for scenario_name, scenario_data in scenarios.items():
        task_data = {
            "set_to_reset": build_same_cycle_pairs(scenario_data),
            "reset_to_next_set": build_following_operation_pairs(
                scenario_data
            ),
        }

        for task_name, (X, Y, diagnostics) in task_data.items():
            current_summary, current_devices = evaluate_models(
                models[task_name],
                X,
                Y,
                scenario_name,
                task_name,
            )
            summary_rows.extend(current_summary)
            device_rows.extend(current_devices)
            diagnostic_rows.append(
                {
                    "scenario": scenario_name,
                    "task": task_name,
                    "devices": X.index.get_level_values("Device").nunique(),
                    **diagnostics,
                }
            )
            shift_rows.extend(
                distribution_shift_rows(
                    training_splits[task_name]["X_train"],
                    X,
                    scenario_name,
                    task_name,
                )
            )

    alpha_rows = [
        {
            "task": task,
            "ridge_alpha": fitted_results[task]["selected_alpha"],
        }
        for task in fitted_results
    ]
    return {
        "external_file": str(external_path),
        "source_rows_present": int(
            (
                external_data.index.get_level_values("Device")
                == SOURCE_DEVICE
            ).sum()
        ),
        "source_copy_matches": source_copy_matches(
            external_data,
            pd.read_parquet(Path(__file__).with_name("F2s_7_0.parquet")),
        ),
        "alphas": pd.DataFrame(alpha_rows),
        "summary": pd.DataFrame(summary_rows),
        "per_device": pd.DataFrame(device_rows),
        "diagnostics": pd.DataFrame(diagnostic_rows),
        "distribution_shift": pd.DataFrame(shift_rows),
    }


def save_results(result):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for result_name in [
        "alphas",
        "summary",
        "per_device",
        "diagnostics",
        "distribution_shift",
    ]:
        result[result_name].to_csv(
            OUTPUT_DIRECTORY / f"external_device_{result_name}.csv",
            index=False,
        )


def print_results(result):
    print("=== EXTERNAL DEVICE TEST ===")
    print(f"File: {result['external_file']}")
    print(f"Source-device rows found: {result['source_rows_present']}")
    print(
        "Source-device copy exactly matches the original file: "
        f"{result['source_copy_matches']}"
    )
    print("Source device excluded from every external metric.")

    selected = result["summary"].loc[
        result["summary"]["scenario"].eq("matched_condition")
        & result["summary"]["aggregation"].eq("pooled")
        & result["summary"]["model"].eq("ridge")
    ]
    for task_name in ["set_to_reset", "reset_to_next_set"]:
        task_results = selected.loc[selected["task"].eq(task_name)].set_index(
            "target"
        )
        print(f"\n=== MATCHED CONDITION: {task_name.upper()} RIDGE ===")
        print(
            task_results.loc[FEATURES, ["RMSE", "MAE_percent", "R2"]]
            .to_string(float_format=lambda value: f"{value:.6g}")
        )


def parse_arguments():
    parser = ArgumentParser(
        description="Evaluate the frozen linear and Ridge models on new devices."
    )
    parser.add_argument(
        "external_data",
        type=Path,
        help="Path to a parquet file with the original dataset schema.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    external_result = run_external_test(arguments.external_data)
    print_results(external_result)
    save_results(external_result)
