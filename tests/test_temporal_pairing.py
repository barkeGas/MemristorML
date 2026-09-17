import unittest

import numpy as np
import pandas as pd

from reset_to_next_set import FEATURES, build_temporal_pairs


def synthetic_dataset():
    index = pd.MultiIndex.from_tuples(
        [
            (1.0, 100, "device_a", 0),
            (1.0, 100, "device_a", 1),
            (1.0, 100, "device_a", 2),
            (1.0, 100, "device_b", 0),
            (1.0, 100, "device_b", 1),
        ],
        names=["RESET_pw", "RESET_target", "Device", "Cycle"],
    )
    data = pd.DataFrame(
        {
            **{
                (operation, feature): np.zeros(len(index), dtype=float)
                for operation in ["SET", "RESET"]
                for feature in FEATURES
            },
            ("SET", "has_error"): np.zeros(len(index), dtype=bool),
            ("RESET", "has_error"): np.zeros(len(index), dtype=bool),
        },
        index=index,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)

    for row_number, row_index in enumerate(index):
        for feature_number, feature in enumerate(FEATURES):
            data.loc[row_index, ("SET", feature)] = (
                1000 + 10 * row_number + feature_number
            )
            data.loc[row_index, ("RESET", feature)] = (
                2000 + 10 * row_number + feature_number
            )
        data.loc[row_index, ("SET", "has_error")] = False
        data.loc[row_index, ("RESET", "has_error")] = False

    return data


class TemporalPairingTests(unittest.TestCase):
    def test_pairs_are_shifted_only_within_each_device(self):
        data = synthetic_dataset()
        X, Y, pair_info, diagnostics = build_temporal_pairs(data)

        self.assertEqual(diagnostics["valid_pairs"], 3)
        cycle_steps = pair_info["target_cycle"] - pair_info["input_cycle"]
        self.assertTrue(cycle_steps.eq(1).all())

        first_a_target = (1.0, 100, "device_a", 1)
        first_b_target = (1.0, 100, "device_b", 1)

        expected_a_input = data.loc[
            (1.0, 100, "device_a", 0),
            [("RESET", feature) for feature in FEATURES],
        ].to_numpy(dtype=float)
        expected_b_input = data.loc[
            (1.0, 100, "device_b", 0),
            [("RESET", feature) for feature in FEATURES],
        ].to_numpy(dtype=float)

        np.testing.assert_array_equal(
            X.loc[first_a_target].to_numpy(dtype=float),
            expected_a_input,
        )
        np.testing.assert_array_equal(
            X.loc[first_b_target].to_numpy(dtype=float),
            expected_b_input,
        )

        expected_a_target = data.loc[
            first_a_target,
            [("SET", feature) for feature in FEATURES],
        ].to_numpy(dtype=float)
        np.testing.assert_array_equal(
            Y.loc[first_a_target].to_numpy(dtype=float),
            expected_a_target,
        )

    def test_missing_values_and_error_flags_remove_candidates(self):
        data = synthetic_dataset()
        data.loc[
            (1.0, 100, "device_a", 2),
            ("SET", "final_v"),
        ] = np.nan
        data.loc[
            (1.0, 100, "device_b", 0),
            ("RESET", "has_error"),
        ] = True

        X, Y, _, diagnostics = build_temporal_pairs(data)

        self.assertEqual(diagnostics["with_nan"], 1)
        self.assertEqual(diagnostics["with_error"], 1)
        self.assertEqual(diagnostics["valid_pairs"], 1)
        self.assertEqual(list(X.index.get_level_values("Device")), ["device_a"])
        self.assertEqual(list(Y.index.get_level_values("Cycle")), [1])


if __name__ == "__main__":
    unittest.main()
