import unittest

import numpy as np
import pandas as pd

from factors.base.reversal_momentum import calc_consec_up, calc_max_drawdown


class ReversalMomentumFixTest(unittest.TestCase):
    def test_consecutive_up_counts_only_trailing_run(self):
        df = pd.DataFrame({
            "symbol": ["A"] * 7,
            "close": [10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 14.0],
        })
        result = calc_consec_up(df, 3)
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertEqual(result.iloc[3], 0.0)
        self.assertEqual(result.iloc[4], -1.0)
        self.assertEqual(result.iloc[6], -3.0)

    def test_max_drawdown_obeys_peak_before_trough(self):
        # 最低价先出现、最高价后出现时，不应被误算成大回撤。
        rising = pd.Series([8.0, 9.0, 10.0])
        self.assertEqual(calc_max_drawdown(rising, 3).iloc[-1], 0.0)

        falling = pd.Series([10.0, 8.0, 9.0])
        self.assertAlmostEqual(calc_max_drawdown(falling, 3).iloc[-1], 0.2)


if __name__ == "__main__":
    unittest.main()
