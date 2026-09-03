"""Run the frozen weekly candidate-factor backtest on the training period only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.run_backtest import _attach_open_limit_state
from backtest.weekly_factor_engine import run_weekly_factor_backtest


CONFIG = ROOT / "config/quantmind_weekly_backtest.json"
AUDIT = ROOT / "reports/quantmind_trials/20260901T064435Z/candidate_sota_audit"
OUT = ROOT / "reports/quantmind_trials/20260901T064435Z/weekly_backtest"
SCORE = "QM_DS_LIQUIDITY_AMPLIFICATION_TURNOVER_RETVOL_WINSOR_Z_SIZE_NEUTRAL"


def main() -> None:
    parser = argparse.ArgumentParser(description="QUANTMIND候选因子通用周频训练期回测")
    parser.add_argument("--signals", type=Path, default=AUDIT / "candidate_audited_values.parquet")
    parser.add_argument("--score-col", default=SCORE)
    parser.add_argument("--direction", choices=("positive", "negative"), default="negative")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    start, end = map(pd.Timestamp, cfg["research_period"])
    signals = pd.read_parquet(args.signals, columns=["symbol", "date", args.score_col],
                              filters=[[('date', '>=', start), ('date', '<=', end)]])
    if args.direction == "negative":
        signals[args.score_col] = -signals[args.score_col]
    market_cols = ["symbol", "date", "open", "close", "factor", "volume"]
    market = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                             columns=market_cols,
                             filters=[[('date', '>=', start), ('date', '<=', end)]])
    market = _attach_open_limit_state(market)
    benchmark = pd.read_parquet(ROOT / "data/raw/daily/daily_sh.parquet",
                                columns=["symbol", "date", "close"],
                                filters=[[('symbol', '==', 'SH000300'), ('date', '>=', start), ('date', '<=', end)]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for bps in cfg["one_way_cost_bps"]:
        result = run_weekly_factor_backtest(
            signals, market, benchmark, score_col=args.score_col, start=start, end=end, cost_bps=int(bps),
            long_quantile=cfg["portfolio"]["long_quantile"],
            initial_cash=cfg["portfolio"]["initial_cash"], lot_size=cfg["portfolio"]["lot_size"],
            minimum_cross_section=cfg["portfolio"]["minimum_cross_section"],
        )
        case = args.output_dir / f"cost_{bps}bps"
        case.mkdir(parents=True, exist_ok=True)
        result.nav.to_parquet(case / "nav.parquet", index=False)
        result.trades.to_parquet(case / "trades.parquet", index=False)
        result.holdings.to_parquet(case / "holdings.parquet", index=False)
        result.schedule.to_parquet(case / "weekly_schedule.parquet", index=False)
        summaries.append({"cost_bps": int(bps), **result.metrics})
        print(f"completed {bps} bps", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "cost_scenario_summary.csv", index=False, encoding="utf-8-sig")
    primary = summary.loc[summary["cost_bps"] == cfg["primary_cost_bps"]].iloc[0]
    gates = cfg["admission_reference"]
    passed = (primary["sharpe"] >= gates["minimum_net_sharpe"] and
              primary["annual_excess_return"] >= gates["minimum_net_annual_excess_return"] and
              primary["max_drawdown"] >= -gates["maximum_net_drawdown_abs"])
    report = {
        "status": "weekly_training_backtest_complete", "candidate": args.score_col,
        "signal_file": str(args.signals), "direction": args.direction,
        "period": cfg["research_period"], "validation_rows_read": 0, "test_rows_read": 0,
        "schedule": cfg["schedule"], "portfolio": cfg["portfolio"],
        "cost_scenarios": summaries, "primary_cost_bps": cfg["primary_cost_bps"],
        "training_reference_gates": gates, "training_gates_passed": bool(passed),
        "formal_sota_decision": "eligible_for_next_frozen_step" if passed else "remain_experimental",
        "warning": "Training backtest is not final proof. Validation was not reused and test was not accessed."
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
