from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factors.base.gp_huatai_replication import (
    FACTOR_NAMES,
    calc_gp_ht_alpha1,
    calc_gp_ht_alpha2,
    calc_gp_ht_alpha3,
    calc_gp_ht_alpha4,
    calc_gp_ht_alpha5,
    calc_gp_ht_alpha6,
)
from factors.registry import get_factor


class HuataiGPReplicationTests(unittest.TestCase):
    @staticmethod
    def frame(days: int = 45, symbols: int = 8) -> pd.DataFrame:
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=days)
        for si in range(symbols):
            for di, date in enumerate(dates):
                close = 10 + si * 0.4 + di * (0.01 + si * 0.001)
                high = close * (1.01 + 0.0002 * ((di + si) % 5))
                low = close * (0.99 - 0.0001 * ((2 * di + si) % 4))
                volume = 1000 + si * 100 + di * (8 + si) + (di % 3) * 7
                vwap = (high + low + close) / 3
                rows.append({"symbol": f"S{si:02d}", "date": date,
                             "close": close, "high": high, "low": low,
                             "volume": volume, "amount": vwap * volume,
                             "vwap": vwap})
        return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)

    def test_all_six_are_registered(self):
        self.assertTrue(all(get_factor(name) is not None for name in FACTOR_NAMES))

    def test_formulas_produce_finite_values_after_warmup(self):
        frame = self.frame()
        functions = (calc_gp_ht_alpha1, calc_gp_ht_alpha2, calc_gp_ht_alpha3,
                     calc_gp_ht_alpha4, calc_gp_ht_alpha5, calc_gp_ht_alpha6)
        for function in functions:
            result = function(frame)
            self.assertEqual(len(result), len(frame))
            self.assertGreater(np.isfinite(result).sum(), 0, function.__name__)

    def test_alpha3_matches_negative_five_day_volume_std(self):
        frame = self.frame(symbols=2)
        expected = -frame.groupby("symbol")["volume"].transform(
            lambda s: s.rolling(5, min_periods=5).std()
        )
        pd.testing.assert_series_equal(
            calc_gp_ht_alpha3(frame), expected, check_names=False
        )

    def test_future_rows_do_not_change_historical_values(self):
        frame = self.frame(days=45)
        cutoff = pd.Timestamp("2024-02-15")
        old = frame[frame["date"] <= cutoff].copy()
        functions = (calc_gp_ht_alpha1, calc_gp_ht_alpha2, calc_gp_ht_alpha3,
                     calc_gp_ht_alpha4, calc_gp_ht_alpha5, calc_gp_ht_alpha6)
        for function in functions:
            full_result = function(frame).loc[old.index]
            old_result = function(old)
            pd.testing.assert_series_equal(full_result, old_result, check_names=False)

    def test_alpha6_report_bracket_interpretation(self):
        frame = self.frame(symbols=2)
        ratio = (frame["high"] + frame["low"]) / (frame["close"] + 1e-12)
        expected = ratio.groupby(frame["symbol"]).transform(
            lambda s: s.rolling(5, min_periods=5).sum()
        )
        pd.testing.assert_series_equal(
            calc_gp_ht_alpha6(frame), expected, check_names=False,
            rtol=1e-12, atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
