from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.alpha_signal import neutralize_model_score, select_with_buffer
from backtest.config import PortfolioConfig
from models.lightgbm.config import LABEL_COL
from backtest.engine import _tradable, run_portfolio_backtest
from backtest.run_backtest import _attach_open_limit_state
from models.lightgbm.data import build_rolling_windows, get_feature_cols, label_horizon


class TimeSplitTests(unittest.TestCase):
    def test_default_label_matches_weekly_rebalance(self):
        self.assertEqual(LABEL_COL, "label_ret_5")
        self.assertEqual(PortfolioConfig().rebalance_days, 5)
        self.assertEqual(PortfolioConfig().max_drop, 10)
        self.assertEqual(PortfolioConfig().max_rebalance_turnover, 0.15)

    def test_label_horizon(self):
        self.assertEqual(label_horizon("label_ret_20"), 20)
        with self.assertRaises(ValueError):
            label_horizon("target")

    def test_purged_windows_do_not_overlap_forward_label(self):
        dates = np.arange(200)
        horizon = 20
        windows = build_rolling_windows(
            dates, {"train_len": 60, "valid_len": 30, "step": 20}, purge_days=horizon
        )
        self.assertTrue(windows)
        for train, valid, test in windows:
            self.assertLess(train[-1] + horizon, valid[0])
            self.assertLess(valid[-1] + horizon, test[0])

    def test_no_forward_label_can_become_feature(self):
        frame = pd.DataFrame(columns=["symbol", "date", "factor_a",
                                      "label_ret_5", "label_ret_10", "label_ret_20"])
        self.assertEqual(get_feature_cols(frame, "label_ret_20"), ["factor_a"])


class SignalTests(unittest.TestCase):
    def _frame(self, n=100):
        rng = np.random.default_rng(42)
        industries = np.array([f"I{i % 10}" for i in range(n)])
        mv = np.exp(rng.normal(10, 1, n))
        pred = 0.8 * np.log(mv) + np.array([int(x[1:]) for x in industries]) + rng.normal(0, 0.2, n)
        return pd.DataFrame({"symbol": [f"S{i:03d}" for i in range(n)],
                             "date": pd.Timestamp("2024-01-02"), "pred": pred,
                             "industry": industries, "circ_mv": mv})

    def test_output_neutralization(self):
        out = neutralize_model_score(self._frame())
        self.assertAlmostEqual(float(out["alpha"].mean()), 0.0, places=8)
        self.assertAlmostEqual(float(out["alpha"].std(ddof=1)), 1.0, places=8)
        self.assertLess(abs(out["alpha"].corr(np.log(out["circ_mv"]))), 1e-8)

    def test_selection_respects_industry_cap(self):
        ranked = neutralize_model_score(self._frame())
        selected = select_with_buffer(ranked, set(), 20, 10, 5, 0.20)
        industries = ranked.set_index("symbol").loc[selected, "industry"].value_counts()
        self.assertEqual(len(selected), 20)
        self.assertLessEqual(int(industries.max()), 4)

    def test_selection_is_independent_of_current_set_iteration_order(self):
        ranked = neutralize_model_score(self._frame())
        symbols = ranked["symbol"].head(20).tolist()
        first = select_with_buffer(ranked, set(symbols), 20, 10, 5, 0.20)
        second = select_with_buffer(ranked, set(reversed(symbols)), 20, 10, 5, 0.20)
        self.assertEqual(first, second)


class EngineTests(unittest.TestCase):
    def test_partial_sell_preserves_inventory_and_nav(self):
        """平价零成本市场中，受预算限制的部分换仓不得凭空损失资产。"""
        dates = pd.bdate_range("2024-01-02", periods=3)
        symbols = [f"S{i:03d}" for i in range(20)]
        market_rows, pred_rows = [], []
        for di, date in enumerate(dates):
            for i, symbol in enumerate(symbols):
                market_rows.append({
                    "symbol": symbol, "date": date, "open": 10.0, "close": 10.0,
                    "volume": 1_000_000, "circ_mv": 1e9, "industry": "I0",
                    "open_at_limit_up": False, "open_at_limit_down": False,
                })
                # 第二个信号日完全反转排序，强制触发预算内的部分卖出。
                score = float(20 - i) if di == 0 else float(i)
                pred_rows.append({
                    "symbol": symbol, "date": date, "pred": score,
                    "industry": "I0", "circ_mv": 1e9,
                })
        benchmark = pd.DataFrame({"date": dates, "close": 100.0})
        cfg = PortfolioConfig(
            topk=10, buffer=0, max_drop=10, rebalance_days=1,
            max_industry_weight=1.0, initial_cash=1_000_000,
            buy_cost=0.0, sell_cost=0.0, min_commission=0.0,
            max_rebalance_turnover=0.02,
        )
        result = run_portfolio_backtest(
            pd.DataFrame(pred_rows), pd.DataFrame(market_rows), benchmark, cfg
        )
        self.assertAlmostEqual(float(result.nav["nav"].iloc[-1]), 1.0, places=10)

    def test_open_limit_state_uses_prior_close_not_full_day_flag(self):
        rows = pd.DataFrame([
            {"symbol": "SH600000", "date": "2024-01-02", "open": 10.0,
             "close": 10.0, "factor": 1.0, "volume": 1_000_000},
            # 开盘未涨停，即使随后收盘涨停，开盘买单仍应允许成交。
            {"symbol": "SH600000", "date": "2024-01-03", "open": 10.2,
             "close": 11.0, "factor": 1.0, "volume": 1_000_000,
             "limit_up": True},
            # 昨收10元、开盘11元，主板开盘涨停，保守判定买不进。
            {"symbol": "SH600001", "date": "2024-01-02", "open": 10.0,
             "close": 10.0, "factor": 1.0, "volume": 1_000_000},
            {"symbol": "SH600001", "date": "2024-01-03", "open": 11.0,
             "close": 11.0, "factor": 1.0, "volume": 1_000_000},
        ])
        marked = _attach_open_limit_state(rows)
        day = marked[marked["date"] == "2024-01-03"].set_index("symbol")
        self.assertTrue(_tradable(day.loc["SH600000"], "buy"))
        self.assertFalse(_tradable(day.loc["SH600001"], "buy"))

    def test_next_day_execution_and_costs(self):
        dates = pd.bdate_range("2024-01-02", periods=6)
        symbols = [f"S{i:03d}" for i in range(30)]
        market_rows, pred_rows = [], []
        for di, date in enumerate(dates):
            for i, symbol in enumerate(symbols):
                market_rows.append({"symbol": symbol, "date": date, "open": 10 + i / 100,
                                    "close": 10.05 + i / 100 + di / 100, "volume": 1_000_000,
                                    "circ_mv": 1e9 + i * 1e7, "industry": f"I{i % 5}",
                                    "limit_up": False, "limit_down": False,
                                    "lock_limit_up": False, "lock_limit_down": False})
                pred_rows.append({"symbol": symbol, "date": date, "pred": float(30 - i),
                                  "industry": f"I{i % 5}", "circ_mv": 1e9 + i * 1e7})
        benchmark = pd.DataFrame({"date": dates, "close": np.linspace(100, 102, len(dates))})
        cfg = PortfolioConfig(topk=10, buffer=5, max_drop=3, rebalance_days=2,
                              max_industry_weight=0.40, initial_cash=1_000_000)
        result = run_portfolio_backtest(pd.DataFrame(pred_rows), pd.DataFrame(market_rows),
                                        benchmark, cfg)
        self.assertFalse(result.nav.empty)
        self.assertFalse(result.trades.empty)
        self.assertGreater(result.metrics["total_cost"], 0)
        self.assertGreater(result.metrics["annualized_cost_ratio"], 0)
        self.assertGreater(result.metrics["initial_build_turnover"], 0)
        self.assertGreater(result.metrics["average_actual_exposure"], 0.80)
        self.assertLessEqual(
            result.metrics["max_rebalance_turnover"],
            cfg.max_rebalance_turnover + 1e-8,
        )
        first_trade_date = pd.Timestamp(result.trades["date"].min())
        self.assertEqual(first_trade_date, dates[1])


if __name__ == "__main__":
    unittest.main()
