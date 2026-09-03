"""复合 Alpha 组合回测入口。

默认只运行验证集。测试集要求显式确认，且成功后写入审计锁，防止反复查看测试集
并据此调参。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.config import PortfolioConfig
from backtest.engine import run_portfolio_backtest
from config.settings import (
    BACKTEST_REPORTS_DIR,
    BASIC_EXTRA_PATH,
    DAILY_RAW_DIR,
    PREDICTIONS_DIR,
    SELECTED_MODEL_PATH,
    TEST_EVALUATION_LOCK,
    TEST_PERIOD,
    VALID_PERIOD,
)


MARKET_COLUMNS = ["symbol", "date", "open", "close", "factor", "volume", "circ_mv", "industry"]


def _limit_rate(symbol: pd.Series, date: pd.Series) -> pd.Series:
    """返回非 ST 股票在给定日期的涨跌停比例。"""
    rate = pd.Series(0.10, index=symbol.index, dtype="float64")
    rate.loc[symbol.str.startswith("SH68")] = 0.20
    gem = symbol.str.startswith("SZ30") & (date >= pd.Timestamp("2020-08-24"))
    rate.loc[gem] = 0.20
    rate.loc[symbol.str.startswith("BJ")] = 0.30
    return rate


def _attach_open_limit_state(market: pd.DataFrame) -> pd.DataFrame:
    """仅用开盘时已知数据构造开盘涨跌停状态。

    Qlib OHLC 为前复权口径，因此先用 factor 还原真实价格，再以此前交易日
    真实收盘价计算交易所涨跌停价。开盘即触及涨/跌停时采用保守假设：对应
    方向无法成交。绝不使用当日收盘后的 pct_chg 或全天 high/low。
    """
    out = market.sort_values(["symbol", "date"]).copy()
    factor = pd.to_numeric(out["factor"], errors="coerce").replace(0, pd.NA)
    out["actual_open"] = pd.to_numeric(out["open"], errors="coerce") / factor
    actual_close = pd.to_numeric(out["close"], errors="coerce") / factor
    out["pre_close"] = actual_close.groupby(out["symbol"], sort=False).shift(1)
    rate = _limit_rate(out["symbol"].astype(str), pd.to_datetime(out["date"]))
    # A股价格按分计价，使用四舍五入到0.01元的理论涨跌停价。
    out["open_limit_price"] = np.floor(out["pre_close"] * (1 + rate) * 100 + 0.5) / 100
    out["open_down_limit_price"] = np.floor(out["pre_close"] * (1 - rate) * 100 + 0.5) / 100
    tolerance = 0.005 + 1e-9
    out["open_at_limit_up"] = (
        out["actual_open"].notna()
        & out["open_limit_price"].notna()
        & (out["actual_open"] >= out["open_limit_price"] - tolerance)
    )
    out["open_at_limit_down"] = (
        out["actual_open"].notna()
        & out["open_down_limit_price"].notna()
        & (out["actual_open"] <= out["open_down_limit_price"] + tolerance)
    )
    return out


def _config_hash(config: PortfolioConfig, model: str, label: str) -> str:
    payload = json.dumps({"model": model, "label": label, **config.to_dict()}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_inputs(model: str, label: str, segment: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period = VALID_PERIOD if segment == "valid" else TEST_PERIOD
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    pred_path = PREDICTIONS_DIR / f"{model}_pred_{label}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"缺少预测文件: {pred_path}")
    pred = pd.read_parquet(
        pred_path,
        filters=[("segment", "==", segment), ("date", ">=", start), ("date", "<=", end)],
    )
    if pred.empty:
        raise ValueError(f"{label} 没有 {segment} 段预测")

    # 多取一个交易日前后缓冲，支持 T+1 执行和首日基准归一化。
    # 风险模型需要足够的信号日前历史行情；这些数据只用于协方差估计。
    market_start, market_end = start - pd.Timedelta(days=240), end + pd.Timedelta(days=10)
    market = pd.read_parquet(
        BASIC_EXTRA_PATH, columns=MARKET_COLUMNS,
        filters=[("date", ">=", market_start), ("date", "<=", market_end)],
    )
    market = _attach_open_limit_state(market)
    pred = pred.merge(market[["symbol", "date", "industry", "circ_mv"]],
                      on=["symbol", "date"], how="inner", validate="many_to_one")

    benchmark_path = DAILY_RAW_DIR / "daily_sh.parquet"
    benchmark = pd.read_parquet(
        benchmark_path, columns=["symbol", "date", "close"],
        filters=[("symbol", "==", "SH000300"), ("date", ">=", market_start),
                 ("date", "<=", market_end)],
    )
    if benchmark.empty:
        raise ValueError("日线数据中缺少沪深300（SH000300）基准")
    return pred, market, benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="复合 Alpha 的 A 股可交易组合回测")
    parser.add_argument("--label", default="label_ret_5",
                        choices=("label_ret_5", "label_ret_10", "label_ret_20"))
    parser.add_argument("--model", default="auto", choices=("auto", "ridge", "lgb", "mlp"),
                        help="auto 读取验证集模型对比生成的 selected_model.json")
    parser.add_argument("--segment", default="valid", choices=("valid", "test"))
    parser.add_argument(
        "--rebalance-days", type=int, default=5,
        help="组合调仓间隔（交易日），与标签期限、模型滚动步长独立；建议比较 5/10/20",
    )
    parser.add_argument(
        "--rebalance-offset", type=int, default=0,
        help="调仓起点偏移；稳健性检验应遍历 0 到 rebalance-days-1",
    )
    parser.add_argument(
        "--max-rebalance-turnover", type=float, default=0.15,
        help="单次调仓双边换手硬上限；默认0.15，即组合权益的15%%",
    )
    parser.add_argument(
        "--dynamic-exposure", action="store_true",
        help="启用波动率/回撤动态仓位；频率研究默认关闭，避免混入风控路径",
    )
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="确认参数已冻结并执行唯一一次测试集评估")
    args = parser.parse_args()
    if args.rebalance_days <= 0:
        parser.error("--rebalance-days 必须为正整数")
    if not 0 <= args.rebalance_offset < args.rebalance_days:
        parser.error("--rebalance-offset 必须满足 0 <= offset < rebalance-days")
    if not 0 < args.max_rebalance_turnover <= 1:
        parser.error("--max-rebalance-turnover 必须满足 0 < value <= 1")

    if args.model == "auto":
        if not SELECTED_MODEL_PATH.exists():
            parser.error("缺少 selected_model.json，请先运行 compare_models")
        selected = json.loads(SELECTED_MODEL_PATH.read_text(encoding="utf-8"))
        args.model = selected["model"]
        if selected.get("label") != args.label:
            parser.error(
                f"selected_model.json 对应 {selected.get('label')}，当前回测要求 {args.label}；"
                f"请先运行 compare_models --label {args.label}"
            )

    if args.segment == "test":
        if not args.confirm_final_test:
            parser.error("测试集只允许最终评估；请在参数冻结后传入 --confirm-final-test")
        if TEST_EVALUATION_LOCK.exists():
            parser.error(f"测试集已评估并锁定，审计记录: {TEST_EVALUATION_LOCK}")

    config = replace(
        PortfolioConfig(),
        rebalance_days=args.rebalance_days,
        rebalance_offset=args.rebalance_offset,
        max_rebalance_turnover=args.max_rebalance_turnover,
        use_dynamic_exposure=args.dynamic_exposure,
    )
    pred, market, benchmark = _load_inputs(args.model, args.label, args.segment)
    result = run_portfolio_backtest(pred, market, benchmark, config)

    # 不同调仓频率分目录保存，避免周频/月频实验互相覆盖。
    out_dir = (BACKTEST_REPORTS_DIR / args.segment / args.model / args.label
               / f"rebalance_{args.rebalance_days}d_offset_{args.rebalance_offset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.nav.to_parquet(out_dir / "nav.parquet", index=False)
    result.trades.to_parquet(out_dir / "trades.parquet", index=False)
    result.holdings.to_parquet(out_dir / "holdings.parquet", index=False)
    summary = {"segment": args.segment, "model": args.model, "label": args.label,
               "config": config.to_dict(), "config_hash": _config_hash(config, args.model, args.label),
               "metrics": result.metrics}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.segment == "test":
        TEST_EVALUATION_LOCK.parent.mkdir(parents=True, exist_ok=True)
        TEST_EVALUATION_LOCK.write_text(json.dumps({
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "model": args.model,
            "config_hash": summary["config_hash"],
            "summary": str(out_dir / "summary.json"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] 回测结果: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
