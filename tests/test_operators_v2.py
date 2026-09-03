import unittest
import numpy as np
import pandas as pd
from data_catalog.operator_audit import audit
from factors import operators_v2 as op
from factors.operators_v2 import apply_by_symbol, cs_rank, cs_zscore, safe_div, ts_delay, ts_pct_change_safe, ts_position, ts_rank_pct, ts_rsquare


class SafeOperatorsTest(unittest.TestCase):
    def test_catalog_and_implementation_match(self): self.assertEqual(audit()["errors"], [])

    def test_pct_change_does_not_bridge_nan(self):
        result = ts_pct_change_safe(pd.Series([1.0, np.nan, 2.0]), 1)
        self.assertTrue(result.iloc[1:].isna().all())

    def test_full_window_and_delay(self):
        values = pd.Series([1.0, 2.0, 3.0])
        self.assertTrue(ts_rank_pct(values, 3).iloc[:2].isna().all())
        self.assertEqual(ts_delay(values, 1).iloc[1], 1.0)

    def test_position_and_rank_are_distinct(self):
        values = pd.Series([1.0, 3.0, 2.0])
        self.assertAlmostEqual(ts_position(values, 3).iloc[-1], 0.5)
        self.assertAlmostEqual(ts_rank_pct(values, 3).iloc[-1], 2 / 3)

    def test_safe_div(self):
        result = safe_div(pd.Series([1.0, 1.0, np.nan]), pd.Series([0.0, 2.0, 3.0]))
        self.assertTrue(np.isnan(result.iloc[0])); self.assertEqual(result.iloc[1], 0.5); self.assertTrue(np.isnan(result.iloc[2]))

    def test_symbol_boundary(self):
        result = apply_by_symbol(pd.Series([1.0, 2.0, 100.0, 200.0]), pd.Series(["A", "A", "B", "B"]), ts_pct_change_safe, 1, dates=pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"])))
        self.assertTrue(np.isnan(result.iloc[0]) and np.isnan(result.iloc[2])); self.assertEqual(result.iloc[1], 1.0); self.assertEqual(result.iloc[3], 1.0)

    def test_symbol_time_must_be_sorted(self):
        values = pd.Series([1.0, 2.0]); symbols = pd.Series(["A", "A"])
        dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-01"]))
        with self.assertRaises(ValueError):
            apply_by_symbol(values, symbols, ts_pct_change_safe, 1, dates=dates)

    def test_safe_div_supports_scalar_and_rejects_index_mismatch(self):
        values = pd.Series([2.0, 4.0], index=[10, 11])
        self.assertEqual(safe_div(values, 2.0).tolist(), [1.0, 2.0])
        self.assertEqual(safe_div(8.0, values).tolist(), [4.0, 2.0])
        with self.assertRaises(ValueError):
            safe_div(values, pd.Series([1.0, 2.0], index=[0, 1]))

    def test_cross_sectional_date_boundary(self):
        values = pd.Series([1.0, 3.0, 10.0, 20.0]); dates = pd.Series(["d1", "d1", "d2", "d2"])
        self.assertEqual(cs_rank(values, dates).tolist(), [0.5, 1.0, 0.5, 1.0])
        self.assertAlmostEqual(float(cs_zscore(values, dates).groupby(dates).mean().abs().max()), 0.0)

    def test_rsquare_constant_is_nan(self):
        self.assertTrue(np.isnan(ts_rsquare(pd.Series([1.0]*3), pd.Series([1.0,2.0,3.0]), 3).iloc[-1]))

    def test_corr_rejects_misaligned_index(self):
        from factors.operators_v2 import ts_corr
        with self.assertRaises(ValueError): ts_corr(pd.Series([1.0,2.0]), pd.Series([1.0,2.0], index=[1,2]), 2)

    def test_safe_math_domains_and_where(self):
        values = pd.Series([-2.0, -1.0, 0.0, 3.0, np.nan])
        self.assertTrue(np.isnan(op.log1p_safe(values).iloc[0]))
        self.assertTrue(np.isnan(op.log1p_safe(values).iloc[1]))
        self.assertEqual(op.sqrt_safe(values).iloc[2], 0.0)
        self.assertTrue(np.isnan(op.sqrt_safe(values).iloc[0]))
        condition = pd.Series([True, False, pd.NA], dtype="boolean")
        result = op.where_values(condition, 1.0, -1.0)
        self.assertEqual(result.iloc[0], 1.0)
        self.assertEqual(result.iloc[1], -1.0)
        self.assertTrue(np.isnan(result.iloc[2]))

    def test_cov_mad_and_ewm_gap_policy(self):
        left = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
        right = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        self.assertAlmostEqual(op.ts_cov(left, right, 3).iloc[2], 2.0)
        self.assertAlmostEqual(op.ts_mad(left, 3).iloc[2], 2.0 / 3.0)
        ewm = op.ewm_mean(left, 3)
        self.assertTrue(np.isnan(ewm.iloc[4]))
        self.assertTrue(np.isnan(ewm.iloc[5]))
        self.assertFalse(np.isnan(ewm.iloc[6]))

    def test_count_and_consecutive_count(self):
        condition = pd.Series([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(op.ts_count(condition, 3).iloc[2], 2.0)
        consecutive = op.ts_consecutive_count(condition, 3)
        self.assertEqual(consecutive.iloc[2], 0.0)
        self.assertEqual(consecutive.iloc[-1], 3.0)
        with self.assertRaises(ValueError):
            op.ts_count(pd.Series([0.0, 2.0]), 2)

    def test_financial_report_lag_uses_distinct_periods_and_is_causal(self):
        values = pd.Series([10.0, 11.0, 20.0, 20.0, 30.0])
        periods = pd.Series(pd.to_datetime([
            "2020-03-31", "2020-03-31", "2020-06-30", "2020-06-30", "2020-09-30"
        ]))
        lagged = op.fin_lag_report(values, periods, 1)
        delta = op.fin_delta_report(values, periods, 1)
        self.assertTrue(lagged.iloc[:2].isna().all())
        self.assertEqual(lagged.iloc[2], 11.0)
        self.assertEqual(lagged.iloc[4], 20.0)
        self.assertEqual(delta.iloc[2], 9.0)

    def test_financial_report_lag_preserves_disclosed_missing_period(self):
        values = pd.Series([10.0, np.nan, 30.0])
        periods = pd.Series(pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"]))
        lagged = op.fin_lag_report(values, periods, 1)
        self.assertEqual(lagged.iloc[1], 10.0)
        self.assertTrue(np.isnan(lagged.iloc[2]))
