"""Backtest frozen LightGBM scores with the project's full TopKDropout engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.config import PortfolioConfig
from backtest.engine import run_portfolio_backtest
from backtest.run_backtest import MARKET_COLUMNS, _attach_open_limit_state

SCORES = ROOT / "reports/lightgbm/human25_qm5_weekly_v2_pit_purged/validation_holdout_scores.parquet"
OUT = ROOT / "reports/lightgbm/human25_qm5_weekly_v2_pit_purged/topk50_buffer30_drop10_10bps_holdout"
START, END = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")


def main() -> int:
    scores = pd.read_parquet(SCORES).rename(columns={"lgb_score": "pred"})
    scores["date"] = pd.to_datetime(scores["date"])
    scores = scores[(scores["date"] >= START) & (scores["date"] <= END)]
    market = pd.read_parquet(
        ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
        columns=MARKET_COLUMNS,
        filters=[[('date', '>=', START - pd.Timedelta(days=240)),
                  ('date', '<=', END + pd.Timedelta(days=10))]],
    )
    market["date"] = pd.to_datetime(market["date"])
    market = _attach_open_limit_state(market)
    pit_industry = pd.read_parquet(
        ROOT / "data/processed/industry/industry_atoms_daily.parquet",
        columns=["symbol", "date", "industry_code"],
        filters=[[('date', '>=', START), ('date', '<=', END)]],
    ).rename(columns={"industry_code": "pit_industry"})
    pit_industry["date"] = pd.to_datetime(pit_industry["date"])
    market = market.merge(pit_industry, on=["symbol", "date"], how="left", validate="one_to_one")
    predictions = scores.merge(
        market[["symbol", "date", "pit_industry", "circ_mv"]]
        .rename(columns={"pit_industry": "industry"}),
        on=["symbol", "date"], how="inner", validate="one_to_one",
    )
    benchmark = pd.read_parquet(
        ROOT / "data/raw/daily/daily_sh.parquet", columns=["symbol", "date", "close"],
        filters=[[('symbol', '==', 'SH000300'),
                  ('date', '>=', START - pd.Timedelta(days=10)),
                  ('date', '<=', END + pd.Timedelta(days=10))]],
    )
    config = PortfolioConfig(
        topk=50, buffer=30, max_drop=10,
        rebalance_rule="calendar_week_end",
        rebalance_days=5, rebalance_offset=0,
        buy_cost=0.001, sell_cost=0.001,
        # Disable the notional turnover cap; TopK+buffer+dropout is the sole
        # regular turnover control requested for this experiment.
        max_rebalance_turnover=float("inf"),
        use_mean_variance_optimizer=False,
        use_dynamic_exposure=False,
    )
    result = run_portfolio_backtest(predictions, market, benchmark, config)
    if result.nav["is_rebalance"].sum() == 0 or result.trades.empty:
        raise RuntimeError("完整周频回测没有产生调仓/成交，拒绝写入无效报告")
    OUT.mkdir(parents=True, exist_ok=True)
    result.nav.to_parquet(OUT / "nav.parquet", index=False)
    result.trades.to_parquet(OUT / "trades.parquet", index=False)
    result.holdings.to_parquet(OUT / "holdings.parquet", index=False)
    config_record = config.to_dict()
    config_record["max_rebalance_turnover"] = "disabled"
    report = {
        "status": "complete", "model": "lightgbm_human25_qm5_v2_pit_purged",
        "score_source": str(SCORES.relative_to(ROOT)).replace("\\", "/"),
        "period": [str(START.date()), str(END.date())],
        "config": config_record, "metrics": result.metrics,
        "test_data_used": False,
        "warning": "Validation holdout result; final test remains locked.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
