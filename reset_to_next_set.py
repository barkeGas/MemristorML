from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("F2s_7_0.parquet")
FEATURES = ["start_v", "final_v", "initial_g", "final_g"]
GROUP_LEVELS = ["RESET_pw", "RESET_target", "Device"]


def load_data():
    return pd.read_parquet(DATA_PATH)


def build_temporal_pairs(data):
    reset_data = data.loc[
        :, [("RESET", feature) for feature in FEATURES]
    ].copy()
    set_data = data.loc[
        :, [("SET", feature) for feature in FEATURES]
    ].copy()

    reset_data.columns = FEATURES
    set_data.columns = FEATURES

    reset_has_error = data.loc[:, ("RESET", "has_error")].copy()
    set_has_error = data.loc[:, ("SET", "has_error")].copy()

    target_cycles = pd.Series(
        data.index.get_level_values("Cycle"),
        index=data.index,
        name="target_cycle",
    )
    input_cycles = target_cycles.groupby(
        level=GROUP_LEVELS,
        sort=False,
    ).shift(1)

    X = reset_data.groupby(
        level=GROUP_LEVELS,
        sort=False,
    ).shift(1)
    Y = set_data.copy()

    input_has_error = reset_has_error.groupby(
        level=GROUP_LEVELS,
        sort=False,
    ).shift(1)

    has_previous_reset = input_cycles.notna()
    is_consecutive = target_cycles.eq(input_cycles + 1)
    candidate_pairs = has_previous_reset & is_consecutive

    has_nan = X.isna().any(axis=1) | Y.isna().any(axis=1)
    has_error = (
        input_has_error.isna()
        | input_has_error.eq(True)
        | set_has_error.isna()
        | set_has_error.eq(True)
    )
    valid_pairs = candidate_pairs & ~has_nan & ~has_error

    diagnostics = {
        "rows": len(data),
        "groups": data.groupby(level=GROUP_LEVELS).ngroups,
        "without_previous_reset": int((~has_previous_reset).sum()),
        "nonconsecutive": int(
            (has_previous_reset & ~is_consecutive).sum()
        ),
        "with_nan": int((candidate_pairs & has_nan).sum()),
        "with_error": int((candidate_pairs & has_error).sum()),
        "valid_pairs": int(valid_pairs.sum()),
    }

    X = X.loc[valid_pairs].copy()
    Y = Y.loc[valid_pairs].copy()
    pair_info = pd.DataFrame(
        {
            "input_cycle": input_cycles.loc[valid_pairs].astype(int),
            "target_cycle": target_cycles.loc[valid_pairs].astype(int),
        },
        index=X.index,
    )

    if not X.index.equals(Y.index):
        raise ValueError("RESET inputs and SET targets do not have matching indices.")

    cycle_steps = pair_info["target_cycle"] - pair_info["input_cycle"]
    if not cycle_steps.eq(1).all():
        raise ValueError("A pair of nonconsecutive cycles was found.")

    return X, Y, pair_info, diagnostics


def chronological_split(X, Y, pair_info):
    if not pair_info["target_cycle"].is_monotonic_increasing:
        raise ValueError("Target cycles are not sorted chronologically.")

    train_end = int(len(X) * 0.80)
    validation_end = int(len(X) * 0.90)

    return {
        "X_train": X.iloc[:train_end].copy(),
        "Y_train": Y.iloc[:train_end].copy(),
        "pairs_train": pair_info.iloc[:train_end].copy(),
        "X_validation": X.iloc[train_end:validation_end].copy(),
        "Y_validation": Y.iloc[train_end:validation_end].copy(),
        "pairs_validation": pair_info.iloc[
            train_end:validation_end
        ].copy(),
        "X_test": X.iloc[validation_end:].copy(),
        "Y_test": Y.iloc[validation_end:].copy(),
        "pairs_test": pair_info.iloc[validation_end:].copy(),
    }


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
                "MAE_percent": 100 * mae / mean_absolute_target,
                "R2": r2_score(
                    y_true[target],
                    y_predicted[target],
                ),
            }
        )

    return pd.DataFrame(rows).set_index("target")


def run_experiment():
    data = load_data()
    X, Y, pair_info, diagnostics = build_temporal_pairs(data)
    result = {
        "X": X,
        "Y": Y,
        "pair_info": pair_info,
        "diagnostics": diagnostics,
    }
    result.update(chronological_split(X, Y, pair_info))

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(result["X_train"], result["Y_train"])

    validation_predictions = pd.DataFrame(
        model.predict(result["X_validation"]),
        index=result["Y_validation"].index,
        columns=FEATURES,
    )
    test_predictions = pd.DataFrame(
        model.predict(result["X_test"]),
        index=result["Y_test"].index,
        columns=FEATURES,
    )

    result.update(
        {
            "model": model,
            "validation_predictions": validation_predictions,
            "test_predictions": test_predictions,
            "validation_metrics": regression_metrics(
                result["Y_validation"],
                validation_predictions,
            ),
            "test_metrics": regression_metrics(
                result["Y_test"],
                test_predictions,
            ),
        }
    )

    return result


def cycle_range(pair_info):
    input_start = pair_info["input_cycle"].iloc[0]
    input_end = pair_info["input_cycle"].iloc[-1]
    target_start = pair_info["target_cycle"].iloc[0]
    target_end = pair_info["target_cycle"].iloc[-1]
    return (
        f"RESET {input_start}-{input_end} -> "
        f"SET {target_start}-{target_end}"
    )


def print_results(result):
    diagnostics = result["diagnostics"]

    print("=== BUILDING RESET_n -> SET_n+1 PAIRS ===")
    print(f"Number of rows: {diagnostics['rows']}")
    print(f"Number of groups: {diagnostics['groups']}")
    print(
        "Rows without a previous RESET cycle: "
        f"{diagnostics['without_previous_reset']}"
    )
    print(
        "Nonconsecutive cycle pairs: "
        f"{diagnostics['nonconsecutive']}"
    )
    print(f"Pairs with NaN values: {diagnostics['with_nan']}")
    print(f"Pairs with errors: {diagnostics['with_error']}")
    print(f"Valid pairs: {diagnostics['valid_pairs']}")

    print("\n=== CHRONOLOGICAL 80/10/10 SPLIT ===")
    for split_name in ["train", "validation", "test"]:
        X_split = result[f"X_{split_name}"]
        Y_split = result[f"Y_{split_name}"]
        pair_split = result[f"pairs_{split_name}"]
        print(
            f"{split_name.capitalize():<11} "
            f"X{X_split.shape}, Y{Y_split.shape}, "
            f"{cycle_range(pair_split)}"
        )

    print("\n=== VALIDATION METRICS ===")
    print(
        result["validation_metrics"].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
    )

    print("\n=== TEST METRICS ===")
    print(
        result["test_metrics"].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
    )


if __name__ == "__main__":
    experiment_result = run_experiment()
    print_results(experiment_result)
