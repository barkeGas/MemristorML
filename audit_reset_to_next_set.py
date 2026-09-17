import numpy as np
import pandas as pd

from reset_to_next_set import (
    DATA_PATH,
    FEATURES,
    GROUP_LEVELS,
    build_temporal_pairs,
    chronological_split,
)


def build_pairs_independently(data):
    input_rows = []
    target_rows = []
    pair_rows = []
    target_indices = []
    nan_exclusions = []
    error_exclusions = []
    candidate_pairs = 0
    nonconsecutive_pairs = 0

    for _, group in data.groupby(level=GROUP_LEVELS, sort=False):
        cycles = group.index.get_level_values("Cycle").to_numpy()

        if len(np.unique(cycles)) != len(cycles):
            raise AssertionError("Duplicate cycles were found within a group.")
        if not np.all(np.diff(cycles) > 0):
            raise AssertionError("Cycles are not strictly increasing within a group.")

        for position in range(1, len(group)):
            input_index = group.index[position - 1]
            target_index = group.index[position]
            input_cycle = int(input_index[-1])
            target_cycle = int(target_index[-1])

            if target_cycle != input_cycle + 1:
                nonconsecutive_pairs += 1
                continue

            candidate_pairs += 1
            input_row = group.iloc[position - 1]
            target_row = group.iloc[position]

            reset_values = input_row.loc[
                [("RESET", feature) for feature in FEATURES]
            ]
            set_values = target_row.loc[
                [("SET", feature) for feature in FEATURES]
            ]

            has_nan = reset_values.isna().any() or set_values.isna().any()
            has_error = (
                pd.isna(input_row.loc[("RESET", "has_error")])
                or bool(input_row.loc[("RESET", "has_error")])
                or pd.isna(target_row.loc[("SET", "has_error")])
                or bool(target_row.loc[("SET", "has_error")])
            )

            if has_nan:
                nan_exclusions.append((input_cycle, target_cycle))
            if has_error:
                error_exclusions.append((input_cycle, target_cycle))
            if has_nan or has_error:
                continue

            input_rows.append(reset_values.to_numpy(dtype=float))
            target_rows.append(set_values.to_numpy(dtype=float))
            pair_rows.append((input_cycle, target_cycle))
            target_indices.append(target_index)

    index = pd.MultiIndex.from_tuples(
        target_indices,
        names=data.index.names,
    )
    X = pd.DataFrame(input_rows, index=index, columns=FEATURES)
    Y = pd.DataFrame(target_rows, index=index, columns=FEATURES)
    pair_info = pd.DataFrame(
        pair_rows,
        index=index,
        columns=["input_cycle", "target_cycle"],
    )

    diagnostics = {
        "candidate_pairs": candidate_pairs,
        "nonconsecutive_pairs": nonconsecutive_pairs,
        "nan_exclusions": nan_exclusions,
        "error_exclusions": error_exclusions,
    }
    return X, Y, pair_info, diagnostics


def verify_split(split, split_name):
    X = split[f"X_{split_name}"]
    Y = split[f"Y_{split_name}"]
    pairs = split[f"pairs_{split_name}"]

    if not X.index.equals(Y.index) or not X.index.equals(pairs.index):
        raise AssertionError(f"Indices are misaligned in the {split_name} split.")
    if not (pairs["target_cycle"] - pairs["input_cycle"]).eq(1).all():
        raise AssertionError(f"An invalid cycle step exists in the {split_name} split.")
    if not pairs["target_cycle"].is_monotonic_increasing:
        raise AssertionError(f"The {split_name} split is not chronological.")


def print_boundary_pair(pair_info, position, label):
    pair = pair_info.iloc[position]
    print(
        f"{label:<24} "
        f"RESET {pair['input_cycle']} -> SET {pair['target_cycle']}"
    )


def run_audit():
    data = pd.read_parquet(DATA_PATH)
    X, Y, pair_info, diagnostics = build_temporal_pairs(data)
    manual_X, manual_Y, manual_pairs, manual_diagnostics = (
        build_pairs_independently(data)
    )

    pd.testing.assert_frame_equal(X, manual_X, check_exact=True)
    pd.testing.assert_frame_equal(Y, manual_Y, check_exact=True)
    pd.testing.assert_frame_equal(pair_info, manual_pairs, check_exact=True)

    indexed_target_cycles = X.index.get_level_values("Cycle").to_numpy()
    if not np.array_equal(
        indexed_target_cycles,
        pair_info["target_cycle"].to_numpy(),
    ):
        raise AssertionError("Target-cycle indices do not match pair metadata.")

    split = chronological_split(X, Y, pair_info)
    for split_name in ["train", "validation", "test"]:
        verify_split(split, split_name)

    if diagnostics["valid_pairs"] != len(manual_X):
        raise AssertionError("The reported and independently counted pair totals differ.")
    if diagnostics["nonconsecutive"] != manual_diagnostics["nonconsecutive_pairs"]:
        raise AssertionError("The nonconsecutive-pair counts differ.")
    if diagnostics["with_nan"] != len(manual_diagnostics["nan_exclusions"]):
        raise AssertionError("The NaN-exclusion counts differ.")
    if diagnostics["with_error"] != len(manual_diagnostics["error_exclusions"]):
        raise AssertionError("The error-exclusion counts differ.")

    print("=== RESET_n -> SET_n+1 DATA AUDIT ===")
    print(f"Raw dataset rows: {len(data)}")
    print(f"Experimental groups: {diagnostics['groups']}")
    print(f"Consecutive candidate pairs: {manual_diagnostics['candidate_pairs']}")
    print(f"Pairs excluded because of NaN values: {diagnostics['with_nan']}")
    print(f"Pairs excluded because of error flags: {diagnostics['with_error']}")
    print(f"Valid temporal pairs: {diagnostics['valid_pairs']}")
    print("Independent row-by-row reconstruction: PASS")
    print("Exact RESET input comparison: PASS (maximum difference = 0)")
    print("Exact SET target comparison: PASS (maximum difference = 0)")
    print("Every target cycle equals input cycle + 1: PASS")

    excluded_pairs = manual_diagnostics["nan_exclusions"]
    print("\nNaN-excluded RESET -> SET cycle pairs:")
    print(", ".join(f"{input_cycle}->{target_cycle}" for input_cycle, target_cycle in excluded_pairs))

    train_end = len(split["pairs_train"])
    validation_end = train_end + len(split["pairs_validation"])
    print("\nBoundary-pair verification:")
    print_boundary_pair(pair_info, 0, "First valid pair")
    print_boundary_pair(pair_info, train_end - 1, "Last train pair")
    print_boundary_pair(pair_info, train_end, "First validation pair")
    print_boundary_pair(pair_info, validation_end - 1, "Last validation pair")
    print_boundary_pair(pair_info, validation_end, "First test pair")
    print_boundary_pair(pair_info, -1, "Last test pair")

    print("\nSplit sizes:")
    for split_name in ["train", "validation", "test"]:
        pairs = split[f"pairs_{split_name}"]
        print(
            f"{split_name.capitalize():<11} {len(pairs):>4} pairs, "
            f"RESET {pairs['input_cycle'].iloc[0]}-{pairs['input_cycle'].iloc[-1]} "
            f"-> SET {pairs['target_cycle'].iloc[0]}-{pairs['target_cycle'].iloc[-1]}"
        )

    print("\nAUDIT RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    run_audit()
