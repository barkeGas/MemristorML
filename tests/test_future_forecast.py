import unittest

import numpy as np
import pandas as pd

from forecast_future_cycles import FEATURES, forecast_operations, run_forecast


class AffineModel:
    def __init__(self, multiplier, offset):
        self.multiplier = multiplier
        self.offset = offset

    def predict(self, X):
        return X.to_numpy() * self.multiplier + self.offset


def synthetic_data():
    rng = np.random.default_rng(42)
    count = 120
    index = pd.MultiIndex.from_tuples(
        [(1.0, 100, "test_device", cycle) for cycle in range(count)],
        names=["RESET_pw", "RESET_target", "Device", "Cycle"],
    )
    columns = {}
    for operation in ["SET", "RESET"]:
        for feature in FEATURES:
            columns[(operation, feature)] = rng.normal(size=count)
        columns[(operation, "has_error")] = np.zeros(count, dtype=bool)
    data = pd.DataFrame(columns, index=index)
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    return data


class FutureForecastTests(unittest.TestCase):
    def test_recursive_predictions_feed_the_next_step(self):
        forecast = forecast_operations(
            AffineModel(2, 1), AffineModel(3, 2),
            np.ones(4), origin=45, horizon=3,
        )
        self.assertEqual(forecast["RESET"].index.tolist(), [46, 47, 48])
        np.testing.assert_array_equal(
            forecast["SET"].to_numpy(), np.repeat([[3], [23], [143]], 4, axis=1)
        )
        np.testing.assert_array_equal(
            forecast["RESET"].to_numpy(), np.repeat([[11], [71], [431]], 4, axis=1)
        )

    def test_future_values_do_not_change_forecast_or_fitted_models(self):
        data = synthetic_data()
        first = run_forecast(data, horizon=5)
        origin = first["metadata"]["origin_cycle"]
        future = data.index.get_level_values("Cycle") > origin
        changed = data.copy()
        columns = [(operation, feature) for operation in ["SET", "RESET"]
                   for feature in FEATURES]
        changed.loc[future, columns] += 10000
        second = run_forecast(changed, horizon=5, origin_cycle=origin)
        for operation in ["SET", "RESET"]:
            pd.testing.assert_frame_equal(
                first["predicted"][operation], second["predicted"][operation]
            )
            self.assertFalse(first["actual"][operation].equals(second["actual"][operation]))
        for name in ["set_to_reset", "reset_to_next_set"]:
            self.assertLess(first["metadata"][name]["training_target_last_cycle"], origin)

    def test_invalid_future_cycle_is_not_silently_skipped(self):
        data = synthetic_data()
        first = run_forecast(data, horizon=5)
        origin = first["metadata"]["origin_cycle"]
        data.loc[(1.0, 100, "test_device", origin + 2), ("SET", "final_v")] = np.nan
        with self.assertRaisesRegex(ValueError, "No complete"):
            run_forecast(data, horizon=5, origin_cycle=origin)

    def test_multiple_devices_are_rejected(self):
        data = synthetic_data()
        other = data.rename(index={"test_device": "other_device"}, level="Device")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            run_forecast(pd.concat([data, other]), horizon=5)

    def test_horizon_beyond_test_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No complete"):
            run_forecast(synthetic_data(), horizon=40)


if __name__ == "__main__":
    unittest.main()
