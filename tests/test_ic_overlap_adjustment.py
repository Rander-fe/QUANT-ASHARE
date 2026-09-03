import unittest

import numpy as np
import pandas as pd

from models.lightgbm.evaluate import (
    _label_horizon,
    _newey_west_mean_test,
    summarize_ic,
)


class TestIcOverlapAdjustment(unittest.TestCase):
    def test_label_horizon(self):
        self.assertEqual(_label_horizon("label_ret_20"), 20)
        self.assertEqual(_label_horizon("unknown"), 1)

    def test_nonoverlap_uses_all_offsets(self):
        daily = pd.DataFrame({
            "ic": np.linspace(0.01, 0.05, 100),
            "rank_ic": np.linspace(0.02, 0.10, 100),
        })
        result = summarize_ic(daily, horizon=20)
        self.assertEqual(result["nonoverlap_offsets"], 20)
        self.assertEqual(result["nonoverlap_min_observations"], 5)
        self.assertEqual(result["nonoverlap_max_observations"], 5)
        self.assertEqual(result["rank_ic_nonoverlap_positive_ratio"], 1.0)
        self.assertEqual(result["rank_ic_nw_lags"], 19)

    def test_hac_detects_autocorrelation(self):
        rng = np.random.default_rng(7)
        innovations = rng.normal(size=500)
        values = np.empty(500)
        values[0] = innovations[0]
        for i in range(1, len(values)):
            values[i] = 0.9 * values[i - 1] + innovations[i]
        hac_se, _ = _newey_west_mean_test(pd.Series(values), max_lags=19)
        naive_se = pd.Series(values).std(ddof=1) / np.sqrt(len(values))
        self.assertGreater(hac_se, naive_se)


if __name__ == "__main__":
    unittest.main()
