# -*- coding: utf-8 -*-
"""诊断原始模型分数、中性化分数与可投资TopK之间的信号损耗。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.alpha_signal import neutralize_model_score, select_with_buffer
from config.settings import BASIC_EXTRA_PATH, DATA_PROCESSED, PREDICTIONS_DIR, REPORTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="信号到组合映射诊断")
    parser.add_argument("--model", default="lgb")
    parser.add_argument("--label", default="label_ret_5")
    parser.add_argument("--segment", default="valid", choices=("valid",))
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    pred = pd.read_parquet(
        PREDICTIONS_DIR / f"{args.model}_pred_{args.label}.parquet",
        columns=["symbol", "date", "pred", "segment"],
        filters=[("segment", "==", args.segment)],
    )
    dates = pd.to_datetime(pred["date"])
    start, end = dates.min(), dates.max()
    labels = pd.read_parquet(
        DATA_PROCESSED / "labels.parquet",
        columns=["symbol", "date", args.label],
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    market_end = end + pd.Timedelta(days=60)
    meta = pd.read_parquet(
        BASIC_EXTRA_PATH,
        columns=["symbol", "date", "industry", "circ_mv", "open"],
        filters=[("date", ">=", start), ("date", "<=", market_end)],
    )
    meta = meta.sort_values(["symbol", "date"])
    by_symbol = meta.groupby("symbol", sort=False)["open"]
    # T日收盘出信号，T+1开盘建仓；20个信号日后在下一开盘换仓。
    meta["open_to_open_20"] = by_symbol.shift(-21) / by_symbol.shift(-1) - 1.0
    meta = meta[meta["date"] <= end]
    frame = pred.merge(labels, on=["symbol", "date"], how="inner", validate="one_to_one")
    frame = frame.merge(meta, on=["symbol", "date"], how="inner", validate="one_to_one")

    rows = []
    for date, day in frame.groupby("date", sort=True):
        day = day.dropna(subset=["pred", args.label, "industry", "circ_mv"])
        if len(day) < max(args.topk * 2, 100):
            continue
        ranked = neutralize_model_score(day)
        joined = day.merge(ranked[["symbol", "alpha", "rank"]], on="symbol", how="inner")
        selected = select_with_buffer(ranked, set(), args.topk, 0, args.topk, 0.20)
        raw_top = day.nlargest(args.topk, "pred")
        alpha_top = joined[joined["symbol"].isin(selected)]
        rows.append({
            "date": date,
            "n": len(day),
            "raw_rank_ic": day["pred"].corr(day[args.label], method="spearman"),
            "neutralized_rank_ic": joined["alpha"].corr(joined[args.label], method="spearman"),
            "raw_topk_forward_return": raw_top[args.label].mean(),
            "selected_topk_forward_return": alpha_top[args.label].mean(),
            "raw_topk_tradable_return": raw_top["open_to_open_20"].mean(),
            "selected_topk_tradable_return": alpha_top["open_to_open_20"].mean(),
            "label_tradable_return_corr": day[args.label].corr(day["open_to_open_20"]),
            "selection_overlap": len(set(raw_top["symbol"]) & set(selected)) / args.topk,
        })

    daily = pd.DataFrame(rows)
    summary = {
        "model": args.model,
        "label": args.label,
        "segment": args.segment,
        "days": int(len(daily)),
        "raw_rank_ic_mean": float(daily["raw_rank_ic"].mean()),
        "neutralized_rank_ic_mean": float(daily["neutralized_rank_ic"].mean()),
        "raw_topk_forward_return_mean": float(daily["raw_topk_forward_return"].mean()),
        "selected_topk_forward_return_mean": float(daily["selected_topk_forward_return"].mean()),
        "raw_topk_tradable_return_mean": float(daily["raw_topk_tradable_return"].mean()),
        "selected_topk_tradable_return_mean": float(daily["selected_topk_tradable_return"].mean()),
        "label_tradable_return_corr_mean": float(daily["label_tradable_return_corr"].mean()),
        "selection_overlap_mean": float(daily["selection_overlap"].mean()),
        "neutralization_ic_positive_ratio": float((daily["neutralized_rank_ic"] > 0).mean()),
    }
    out_dir = REPORTS_DIR / "signal_diagnostics" / args.model / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "daily_signal_mapping.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] 诊断结果: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
