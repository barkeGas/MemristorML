from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("F2s_7_0.parquet")
FEATURES = ["start_v", "final_v", "initial_g", "final_g"]


data = pd.read_parquet(DATA_PATH)

print("=== DATASET ===")
print(f"File: {DATA_PATH.name}")
print(f"Number of rows: {len(data)}")
print(f"Number of columns: {data.shape[1]}")


X = data.loc[:, [("SET", feature) for feature in FEATURES]].copy()
Y = data.loc[:, [("RESET", feature) for feature in FEATURES]].copy()

X.columns = FEATURES
Y.columns = FEATURES


print("\n=== NaN REVIEW BEFORE FILTERING ===")
print("X (SET):")
print(X.isna().sum())
print("\nY (RESET):")
print(Y.isna().sum())


error_flags = data.loc[
    :, [("SET", "has_error"), ("RESET", "has_error")]
].copy()
error_flags.columns = ["SET_has_error", "RESET_has_error"]

print("\n=== has_error REVIEW ===")
for column in error_flags.columns:
    print(f"{column}=True: {int(error_flags[column].eq(True).sum())}")
    print(f"{column}=NaN:  {int(error_flags[column].isna().sum())}")


has_nan = X.isna().any(axis=1) | Y.isna().any(axis=1)
has_error = error_flags.fillna(True).astype(bool).any(axis=1)
valid_rows = ~(has_nan | has_error)

print("\n=== FILTERING ===")
print(f"Rows with NaN in X or Y: {int(has_nan.sum())}")
print(f"Rows with a SET/RESET error: {int(has_error.sum())}")
print(f"Total rows removed: {int((~valid_rows).sum())}")

X = X.loc[valid_rows].copy()
Y = Y.loc[valid_rows].copy()


print("\n=== FINAL X AND Y ===")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"X and Y have the same index: {X.index.equals(Y.index)}")
print(f"Remaining NaN values in X: {int(X.isna().sum().sum())}")
print(f"Remaining NaN values in Y: {int(Y.isna().sum().sum())}")

print("\nFirst 5 rows of X (SET):")
print(X.head())

print("\nFirst 5 rows of Y (RESET):")
print(Y.head())


cycle_numbers = X.index.get_level_values("Cycle")
if not cycle_numbers.is_monotonic_increasing:
    raise ValueError("Cycles are not sorted chronologically.")

train_end = int(len(X) * 0.80)
validation_end = int(len(X) * 0.90)

X_train = X.iloc[:train_end].copy()
Y_train = Y.iloc[:train_end].copy()

X_validation = X.iloc[train_end:validation_end].copy()
Y_validation = Y.iloc[train_end:validation_end].copy()

X_test = X.iloc[validation_end:].copy()
Y_test = Y.iloc[validation_end:].copy()


def cycle_range(frame):
    cycles = frame.index.get_level_values("Cycle")
    return f"{cycles[0]}-{cycles[-1]}"


print("\n=== CHRONOLOGICAL 80/10/10 SPLIT ===")
print(f"Train:      X{X_train.shape}, Y{Y_train.shape}, cycles {cycle_range(X_train)}")
print(
    f"Validation: X{X_validation.shape}, Y{Y_validation.shape}, "
    f"cycles {cycle_range(X_validation)}"
)
print(f"Test:       X{X_test.shape}, Y{Y_test.shape}, cycles {cycle_range(X_test)}")

print("\n=== SPLIT VERIFICATION ===")
print(
    "Total number of rows is preserved: "
    f"{len(X_train) + len(X_validation) + len(X_test) == len(X)}"
)
print(f"Train X/Y indices aligned: {X_train.index.equals(Y_train.index)}")
print(
    "Validation X/Y indices aligned: "
    f"{X_validation.index.equals(Y_validation.index)}"
)
print(f"Test X/Y indices aligned: {X_test.index.equals(Y_test.index)}")


linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model.fit(X_train, Y_train)

Y_validation_predicted = pd.DataFrame(
    linear_model.predict(X_validation),
    index=Y_validation.index,
    columns=FEATURES,
)
Y_test_predicted = pd.DataFrame(
    linear_model.predict(X_test),
    index=Y_test.index,
    columns=FEATURES,
)


def regression_metrics(y_true, y_predicted):
    rows = []

    for target in FEATURES:
        mse = mean_squared_error(y_true[target], y_predicted[target])
        rows.append(
            {
                "target": target,
                "MAE": mean_absolute_error(
                    y_true[target], y_predicted[target]
                ),
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_true[target], y_predicted[target]),
            }
        )

    return pd.DataFrame(rows).set_index("target")


validation_metrics = regression_metrics(
    Y_validation, Y_validation_predicted
)
test_metrics = regression_metrics(Y_test, Y_test_predicted)

print("\n=== LINEAR REGRESSION: VALIDATION METRICS ===")
print(validation_metrics.to_string(float_format=lambda value: f"{value:.6g}"))

print("\n=== LINEAR REGRESSION: TEST METRICS ===")
print(test_metrics.to_string(float_format=lambda value: f"{value:.6g}"))


sample_positions = np.linspace(
    0, len(X_test) - 1, num=5, dtype=int
)

print("\n=== INDIVIDUAL TEST ===")
for position in sample_positions:
    cycle = X_test.iloc[[position]].index.get_level_values("Cycle")[0]
    comparison = pd.DataFrame(
        {
            "actual_RESET": Y_test.iloc[position],
            "predicted_RESET": Y_test_predicted.iloc[position],
        }
    )
    comparison["error"] = (
        comparison["predicted_RESET"] - comparison["actual_RESET"]
    )
    comparison["absolute_error"] = comparison["error"].abs()

    print(f"\n--- Cycle {cycle} ---")
    print("SET input:")
    print(X_test.iloc[position].to_string(float_format=lambda value: f"{value:.6g}"))
    print("RESET: actual, predicted, and error:")
    print(comparison.to_string(float_format=lambda value: f"{value:.6g}"))


X_random_train, X_random_remaining, Y_random_train, Y_random_remaining = (
    train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )
)
(
    X_random_validation,
    X_random_test,
    Y_random_validation,
    Y_random_test,
) = train_test_split(
    X_random_remaining,
    Y_random_remaining,
    test_size=0.50,
    random_state=42,
    shuffle=True,
)

print("\n=== RANDOM 80/10/10 SPLIT (random_state=42) ===")
print(f"Train:      X{X_random_train.shape}, Y{Y_random_train.shape}")
print(
    f"Validation: X{X_random_validation.shape}, "
    f"Y{Y_random_validation.shape}"
)
print(f"Test:       X{X_random_test.shape}, Y{Y_random_test.shape}")
print(
    "Random train X/Y indices aligned: "
    f"{X_random_train.index.equals(Y_random_train.index)}"
)
print(
    "Random validation X/Y indices aligned: "
    f"{X_random_validation.index.equals(Y_random_validation.index)}"
)
print(
    "Random test X/Y indices aligned: "
    f"{X_random_test.index.equals(Y_random_test.index)}"
)

random_linear_model = make_pipeline(StandardScaler(), LinearRegression())
random_linear_model.fit(X_random_train, Y_random_train)

Y_random_validation_predicted = pd.DataFrame(
    random_linear_model.predict(X_random_validation),
    index=Y_random_validation.index,
    columns=FEATURES,
)
Y_random_test_predicted = pd.DataFrame(
    random_linear_model.predict(X_random_test),
    index=Y_random_test.index,
    columns=FEATURES,
)

random_validation_metrics = regression_metrics(
    Y_random_validation, Y_random_validation_predicted
)
random_test_metrics = regression_metrics(
    Y_random_test, Y_random_test_predicted
)

print("\n=== RANDOM LINEAR REGRESSION: VALIDATION METRICS ===")
print(
    random_validation_metrics.to_string(
        float_format=lambda value: f"{value:.6g}"
    )
)
print("\n=== RANDOM LINEAR REGRESSION: TEST METRICS ===")
print(
    random_test_metrics.to_string(
        float_format=lambda value: f"{value:.6g}"
    )
)


split_comparison = pd.DataFrame(index=FEATURES)
split_comparison["chronological_MAE"] = test_metrics["MAE"]
split_comparison["random_MAE"] = random_test_metrics["MAE"]
split_comparison["MAE_change_percent"] = (
    (random_test_metrics["MAE"] / test_metrics["MAE"] - 1.0) * 100.0
)
split_comparison["chronological_R2"] = test_metrics["R2"]
split_comparison["random_R2"] = random_test_metrics["R2"]
split_comparison["R2_change"] = (
    random_test_metrics["R2"] - test_metrics["R2"]
)

print("\n=== CHRONOLOGICAL VS RANDOM TEST ===")
print(
    split_comparison.to_string(
        float_format=lambda value: f"{value:.6g}"
    )
)
