"""在验证集内部用时间留出法选择组合参数；绝不读取测试集。"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.config import PortfolioConfig
from backtest.engine import run_portfolio_backtest
from backtest.run_backtest import _load_inputs
from config.settings import BACKTEST_REPORTS_DIR


TUNE_END = pd.Timestamp("2024-03-29")
HOLDOUT_START = pd.Timestamp("2024-04-01")

# 小规模、预先声明的组合层候选，避免对验证集做大规模数据挖掘。
# 周频使用10%/15%/20%三档单次双边换手硬上限；正式默认档为15%。
PRESETS = {
    "robust_equal": {"max_rebalance_turnover": 0.15},
    "unmanaged_equal": {"target_annual_volatility": 1.0, "minimum_exposure": 1.0,
                        "drawdown_tiers": (), "max_rebalance_turnover": 0.20},
    "low_turnover_equal": {"max_rebalance_turnover": 0.10},
    "mean_variance": {"use_mean_variance_optimizer": True,
                      "target_annual_volatility": 0.28,
                      "minimum_exposure": 0.50,
                      "max_rebalance_turnover": 0.15,
                      "drawdown_tiers": ((-0.15, 0.85), (-0.25, 0.65), (-0.35, 0.50))},
}


def _score(metrics: dict) -> float:
    """偏好风险调整后收益，对回撤、波动和换手施加连续惩罚。"""
    return float(
        metrics["sharpe"]
        + 0.50 * metrics["information_ratio"]
        + 0.75 * metrics["annual_excess_return"]
        - 0.75 * abs(metrics["max_drawdown"])
        - 0.20 * metrics["annual_volatility"]
        - 2.00 * metrics["average_daily_turnover"]
    )


def _run_period(pred, market, benchmark, config, start=None, end=None):
    mask = pd.Series(True, index=pred.index)
    if start is not None:
        mask &= pred["date"] >= start
    if end is not None:
        mask &= pred["date"] <= end
    return run_portfolio_backtest(pred.loc[mask].copy(), market, benchmark, config)


def main() -> int:
    model, label = "lgb", "label_ret_5"
    pred, market, benchmark = _load_inputs(model, label, "valid")
    rows = []
    configs = {}
    for name, overrides in PRESETS.items():
        config = replace(PortfolioConfig(), **overrides)
        configs[name] = config
        result = _run_period(pred, market, benchmark, config, end=TUNE_END)
        rows.append({"preset": name, "selection_score": _score(result.metrics),
                     **result.metrics})
        print(f"[TUNE] {name}: score={rows[-1]['selection_score']:.4f}")

    comparison = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    chosen = str(comparison.iloc[0]["preset"])
    config = configs[chosen]
    holdout = _run_period(pred, market, benchmark, config, start=HOLDOUT_START)
    full = _run_period(pred, market, benchmark, config)

    out = BACKTEST_REPORTS_DIR / "portfolio_tuning" / model / label
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "tune_comparison.csv", index=False)
    holdout.nav.to_parquet(out / "holdout_nav.parquet", index=False)
    holdout.trades.to_parquet(out / "holdout_trades.parquet", index=False)
    full.nav.to_parquet(out / "full_valid_nav.parquet", index=False)
    full.trades.to_parquet(out / "full_valid_trades.parquet", index=False)
    full.holdings.to_parquet(out / "full_valid_holdings.parquet", index=False)
    summary = {
        "model": model, "label": label, "chosen_preset": chosen,
        "tune_period": [str(pred["date"].min().date()), str(TUNE_END.date())],
        "holdout_period": [str(HOLDOUT_START.date()), str(pred["date"].max().date())],
        "config": config.to_dict(), "tune_metrics": comparison.iloc[0].to_dict(),
        "holdout_metrics": holdout.metrics, "full_valid_metrics": full.metrics,
        "test_set_accessed": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
