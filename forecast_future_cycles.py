"""Forecast 40 future cycles using only one measured RESET as the seed.

Run this file in PyCharm. Both models are trained on their original first 80%
of valid pairs. The first complete window in their common final 10% is used
for evaluation. Future measured SET/RESET values are never fed into the loop.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reset_to_next_set import FEATURES, GROUP_LEVELS, build_temporal_pairs, load_data


FORECAST_HORIZON = 40
ORIGIN_CYCLE = None  # None selects the earliest complete held-out window.
PROJECT_DIRECTORY = Path(__file__).resolve().parent
FIGURE_DIRECTORY = PROJECT_DIRECTORY / "docs" / "figures"
RESULT_DIRECTORY = PROJECT_DIRECTORY / "docs" / "results"


def prepare_data(data):
    if not data.index.is_unique:
        raise ValueError("Duplicate cycle indices were found.")
    if len(data.index.droplevel("Cycle").unique()) != 1:
        raise ValueError("This experiment requires exactly one device/condition group.")
    cycles = data.index.get_level_values("Cycle").to_numpy()
    if not np.all(np.isfinite(cycles)) or not np.all(cycles == cycles.astype(int)):
        raise ValueError("Cycle numbers must be finite integers.")
    data = data.sort_index(level="Cycle").copy()
    operations = {}
    valid = {}
    for operation in ["SET", "RESET"]:
        columns = [(operation, feature) for feature in FEATURES]
        data[columns] = data[columns].replace([np.inf, -np.inf], np.nan)
        operations[operation] = data.loc[:, columns].copy()
        operations[operation].columns = FEATURES
        valid[operation] = (
            operations[operation].notna().all(axis=1)
            & data[(operation, "has_error")].eq(False)
        )
    return data, operations, valid


def fit_training_model(X, Y):
    train_end = int(len(X) * 0.80)
    test_start = int(len(X) * 0.90)
    if train_end < 2 or test_start >= len(X):
        raise ValueError("Not enough valid pairs for an 80/10/10 split.")
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X.iloc[:train_end], Y.iloc[:train_end])
    return model, {
        "training_rows": train_end,
        "training_target_first_cycle": int(Y.index.get_level_values("Cycle")[0]),
        "training_target_last_cycle": int(
            Y.index.get_level_values("Cycle")[train_end - 1]
        ),
        "test_first_cycle": int(Y.index.get_level_values("Cycle")[test_start]),
    }, Y.index[test_start:]


def select_forecast_origin(test_index, valid_index, horizon, origin_cycle=None):
    test_cycles = set(test_index.get_level_values("Cycle").astype(int))
    valid_cycles = set(valid_index.get_level_values("Cycle").astype(int))
    candidates = (
        [int(origin_cycle)] if origin_cycle is not None
        else [cycle - 1 for cycle in sorted(test_cycles)]
    )
    for origin in candidates:
        targets = set(range(origin + 1, origin + horizon + 1))
        if origin in valid_cycles and targets.issubset(test_cycles):
            return origin
    raise ValueError(
        f"No complete {horizon}-cycle held-out window for this origin. "
        "A valid measured seed and consecutive valid future cycles are required."
    )


def forecast_operations(reset_to_set, set_to_reset, initial_reset, origin, horizon):
    """Pure recursive prediction: no future observations are accepted here."""
    current_reset = np.asarray(initial_reset, dtype=float).reshape(1, len(FEATURES))
    predictions = {"SET": [], "RESET": []}
    for step in range(1, horizon + 1):
        predicted_set = reset_to_set.predict(
            pd.DataFrame(current_reset, columns=FEATURES)
        )
        predicted_reset = set_to_reset.predict(
            pd.DataFrame(predicted_set, columns=FEATURES)
        )
        if not np.isfinite(predicted_set).all() or not np.isfinite(predicted_reset).all():
            raise ValueError(f"Non-finite prediction at forecast step {step}.")
        predictions["SET"].append(predicted_set[0].copy())
        predictions["RESET"].append(predicted_reset[0].copy())
        current_reset = predicted_reset

    index = pd.Index(range(origin + 1, origin + horizon + 1), name="Cycle")
    return {
        operation: pd.DataFrame(values, index=index, columns=FEATURES)
        for operation, values in predictions.items()
    }


def evaluate_forecasts(actual, predicted, persistence):
    rows = []
    for operation in ["SET", "RESET"]:
        for method, forecast in [
            ("Recursive linear regression", predicted[operation]),
            ("Persistence", persistence[operation]),
        ]:
            for feature in FEATURES:
                true = actual[operation][feature].to_numpy()
                estimate = forecast[feature].to_numpy()
                error = estimate - true
                mse = float(np.mean(error ** 2))
                mae = float(np.mean(np.abs(error)))
                scale = float(np.mean(np.abs(true)))
                rows.append({
                    "operation": operation,
                    "method": method,
                    "target": feature,
                    "MSE": mse,
                    "RMSE": np.sqrt(mse),
                    "MAE": mae,
                    "MAE_percent": 100 * mae / scale if scale else np.nan,
                    "R2": float(r2_score(true, estimate))
                    if len(true) > 1 and np.ptp(true) > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def run_forecast(data=None, horizon=FORECAST_HORIZON, origin_cycle=ORIGIN_CYCLE):
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("The forecast horizon must be a positive integer.")
    data, operations, valid = prepare_data(load_data() if data is None else data)
    same_cycle_valid = valid["SET"] & valid["RESET"]
    set_to_reset, same_info, same_test = fit_training_model(
        operations["SET"].loc[same_cycle_valid],
        operations["RESET"].loc[same_cycle_valid],
    )
    X, Y, _, _ = build_temporal_pairs(data)
    reset_to_set, next_info, next_test = fit_training_model(X, Y)
    common_test = same_test.intersection(next_test, sort=False)
    origin = select_forecast_origin(
        common_test, data.index[same_cycle_valid], horizon, origin_cycle
    )
    if max(same_info["training_target_last_cycle"],
           next_info["training_target_last_cycle"]) > origin:
        raise ValueError("Training targets must not extend beyond the forecast origin.")

    # Retain true cycle IDs; never close gaps by dropping rows and renumbering.
    by_cycle = {
        operation: values.droplevel(GROUP_LEVELS)
        for operation, values in operations.items()
    }
    initial = {operation: values.loc[origin].copy()
               for operation, values in by_cycle.items()}
    predicted = forecast_operations(
        reset_to_set, set_to_reset, initial["RESET"], origin, horizon
    )
    cycles = predicted["RESET"].index
    # These observations are accessed only AFTER the recursive forecast is complete.
    actual = {operation: values.loc[cycles].copy()
              for operation, values in by_cycle.items()}
    persistence = {
        operation: pd.DataFrame(
            np.repeat(initial[operation].to_numpy()[None, :], horizon, axis=0),
            index=cycles, columns=FEATURES,
        ) for operation in ["SET", "RESET"]
    }
    nonpositive = {
        operation: int((predicted[operation][["initial_g", "final_g"]] <= 0)
                       .to_numpy().sum())
        for operation in ["SET", "RESET"]
    }
    group = data.index.droplevel("Cycle")[0]
    metadata = {
        "dataset": "F2s_7_0.parquet" if group[-1] == "F2s_7_0" else "supplied data",
        "device": str(group[-1]),
        "RESET_pw": float(group[0]),
        "RESET_target": float(group[1]),
        "features": FEATURES,
        "origin_cycle": int(origin),
        "horizon_cycles": int(horizon),
        "first_forecast_cycle": int(cycles[0]),
        "last_forecast_cycle": int(cycles[-1]),
        "model": "StandardScaler + LinearRegression (each direction)",
        "set_to_reset": same_info,
        "reset_to_next_set": next_info,
        "protocol": "Train on first 80% of each task's valid pairs; middle 10% "
                    "unused (no tuning); select earliest complete window in common "
                    "final 10%, unless an origin is specified. No retraining during rollout.",
        "forecast_inputs": "Only measured RESET at the origin, then predicted operations.",
        "nonpositive_predicted_conductances": nonpositive,
        "limitations": "One fixed-origin window, not an average over forecast origins. "
                       "No clipping, future measured inputs, or uncertainty intervals.",
    }
    return {
        "actual": actual, "predicted": predicted, "persistence": persistence,
        "initial": initial, "metadata": metadata,
        "metrics": evaluate_forecasts(actual, predicted, persistence),
    }


def plot_forecast(result, output_path):
    # Plotting is optional for programmatic forecasts and synthetic CI tests.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = result["metadata"]
    origin = info["origin_cycle"]
    cycles = result["actual"]["RESET"].index.to_numpy()
    figure, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    for column, operation in enumerate(["SET", "RESET"]):
        for row, feature in enumerate(["final_g", "final_v", "start_v", "initial_g"]):
            axis = axes[row, column]
            seed = result["initial"][operation][feature]
            axis.plot(cycles, result["actual"][operation][feature],
                      color="#1f77b4", linewidth=1.8, marker=".",
                      markersize=4, label="Measured (comparison only)")
            axis.plot(cycles, result["predicted"][operation][feature],
                      color="#e67300", linewidth=2, label="Recursive forecast")
            axis.plot(cycles, result["persistence"][operation][feature],
                      color="#777777", linestyle=":", linewidth=1.6,
                      label="Last-value baseline")
            axis.scatter([origin], [seed], color="black", marker="D", s=24,
                         label="Last observed cycle", zorder=5)
            axis.axvline(origin + 0.5, color="black", alpha=0.3, linewidth=0.8)
            axis.set_title(f"{operation}: {feature}", fontsize=11)
            unit = "S" if feature.endswith("_g") else "V"
            axis.set_ylabel(f"[{unit}]")
            if unit == "S":
                axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axis.grid(alpha=0.25)
            axis.set_xlim(origin - 0.5, cycles[-1] + 0.5)
    for axis in axes[-1]:
        axis.set_xlabel("Cycle")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, 0.958), frameon=False, fontsize=9)
    figure.suptitle(
        f"{info['device']}: {info['horizon_cycles']}-cycle forecast without future measurements",
        fontsize=15, y=0.987,
    )
    figure.text(
        0.5, 0.015,
        f"Seed: measured RESET at cycle {origin}. Then predicted SET -> predicted RESET, repeated.\n"
        "Both models stay fixed. All plotted future measurements are held out. "
        "Conductances are shown without clipping.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.92), h_pad=1.8)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_results(result):
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    info = result["metadata"]
    stem = f"recursive_forecast_{info['horizon_cycles']}_cycles_origin_{info['origin_cycle']}"
    paths = {
        "figure": FIGURE_DIRECTORY / f"{stem}.png",
        "predictions": RESULT_DIRECTORY / f"{stem}_predictions.csv",
        "metrics": RESULT_DIRECTORY / f"{stem}_metrics.csv",
        "metadata": RESULT_DIRECTORY / f"{stem}_metadata.json",
    }
    rows = []
    for operation in ["SET", "RESET"]:
        for step, cycle in enumerate(result["actual"][operation].index, start=1):
            for feature in FEATURES:
                rows.append({
                    "device": info["device"], "origin_cycle": info["origin_cycle"],
                    "horizon": step, "cycle": cycle, "operation": operation,
                    "feature": feature,
                    "actual": result["actual"][operation].loc[cycle, feature],
                    "predicted": result["predicted"][operation].loc[cycle, feature],
                    "persistence": result["persistence"][operation].loc[cycle, feature],
                })
    pd.DataFrame(rows).to_csv(paths["predictions"], index=False)
    result["metrics"].to_csv(paths["metrics"], index=False)
    paths["metadata"].write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    plot_forecast(result, paths["figure"])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON)
    parser.add_argument("--origin-cycle", type=int, default=ORIGIN_CYCLE)
    args = parser.parse_args()
    result = run_forecast(horizon=args.horizon, origin_cycle=args.origin_cycle)
    info = result["metadata"]
    print(f"Device: {info['device']}")
    print(f"Measured RESET seed: cycle {info['origin_cycle']}")
    print(f"Forecast: cycles {info['first_forecast_cycle']}-{info['last_forecast_cycle']}")
    print("No future measured SET or RESET values enter the forecast loop.")
    for name in ["set_to_reset", "reset_to_next_set"]:
        model_info = info[name]
        print(f"{name}: {model_info['training_rows']} training pairs; "
              f"last training target cycle {model_info['training_target_last_cycle']}")
    print("\nMetrics across the complete forecast window (not averages over origins):")
    print(result["metrics"].to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    for operation, count in info["nonpositive_predicted_conductances"].items():
        if count:
            print(f"WARNING: {count} nonpositive {operation} conductance predictions; not clipped.")
    for name, path in save_results(result).items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
