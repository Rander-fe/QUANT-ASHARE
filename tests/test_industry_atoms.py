import unittest

import numpy as np
import pandas as pd

from quantmind_pipeline.industry_atoms import attach_industry_intervals, build_industry_atoms


class IndustryAtomsTest(unittest.TestCase):
    def test_half_open_membership_interval(self):
        daily = pd.DataFrame({"symbol": ["A"]*3, "date": pd.to_datetime([
            "2020-01-01", "2020-01-02", "2020-01-03"]), "close": [1., 1., 1.]})
        intervals = pd.DataFrame({"symbol": ["A"], "start_date": pd.to_datetime(["2020-01-02"]),
                                  "end_date": pd.to_datetime(["2020-01-03"]),
                                  "industry_code": ["I1"], "industry_name": ["one"]})
        out = attach_industry_intervals(daily, intervals)
        self.assertTrue(pd.isna(out.loc[out.date.eq("2020-01-01"), "industry_code"]).all())
        self.assertEqual(out.loc[out.date.eq("2020-01-02"), "industry_code"].iloc[0], "I1")
        self.assertTrue(pd.isna(out.loc[out.date.eq("2020-01-03"), "industry_code"]).all())

    def test_leave_one_out_industry_return(self):
        dates = pd.date_range("2020-01-01", periods=90, freq="B")
        rows = []
        for j in range(6):
            prices = 100*np.cumprod(1 + np.full(len(dates), .001*(j+1)))
            rows.extend({"symbol": f"S{j}", "date": d, "close": p, "industry_code": "I1"}
                        for d, p in zip(dates, prices))
        out = build_industry_atoms(pd.DataFrame(rows))
        day = out.loc[out.date.eq(dates[-1])].set_index("symbol")
        expected = np.mean([.002, .003, .004, .005, .006])
        self.assertAlmostEqual(day.loc["S0", "industry_ret_1d_loo"], expected, places=10)
        self.assertTrue(day["IND_REL_RET_20"].notna().all())


if __name__ == "__main__": unittest.main()
