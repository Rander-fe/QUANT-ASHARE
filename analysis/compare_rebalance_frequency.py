# -*- coding: utf-8 -*-
"""在统一年化换手预算下比较不同调仓频率及全部起点偏移。

默认仅使用验证集。脚本先调用正式组合回测，再汇总每种频率的中位数、
最差值和 offset 胜率，避免挑选最有利的调仓起点。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BACKTEST_REPORTS_DIR, REPORTS_DIR


METRICS = (
    "annual_excess_return",
    "information_ratio",
    "max_drawdown",
    "average_daily_turnover",
    "annualized_two_sided_turnover",
    "initial_build_turnover",
    "average_actual_exposure",
    "total_cost",
    "total_cost_ratio",
    "annualized_cost_ratio",
)


def _parse_frequencies(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values or any(x <= 0 for x in values):
        raise ValueError("frequencies 必须是逗号分隔的正整数")
    return list(dict.fromkeys(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="调仓频率与起点稳健性对比")
    parser.add_argument("--model", default="lgb", choices=("ridge", "lgb", "mlp"))
    parser.add_argument("--label", default="label_ret_5")
    parser.add_argument("--segment", default="valid", choices=("valid",))
    parser.add_argument("--frequencies", default="20,5,10")
    parser.add_argument("--max-rebalance-turnover", type=float, default=0.15)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    frequencies = _parse_frequencies(args.frequencies)
    project_root = Path(__file__).resolve().parent.parent
    runner = project_root / "backtest" / "run_backtest.py"
    rows: list[dict] = []

    for days in frequencies:
        for offset in range(days):
            summary_path = (
                BACKTEST_REPORTS_DIR / args.segment / args.model / args.label
                / f"rebalance_{days}d_offset_{offset}" / "summary.json"
            )
            if not (args.skip_existing and summary_path.exists()):
                command = [
                    sys.executable, str(runner),
                    "--model", args.model,
                    "--label", args.label,
                    "--segment", args.segment,
                    "--rebalance-days", str(days),
                    "--rebalance-offset", str(offset),
                    "--max-rebalance-turnover", str(args.max_rebalance_turnover),
                ]
                print(f"[RUN] {days}d offset={offset}", flush=True)
                subprocess.run(command, cwd=project_root, check=True)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            rows.append({
                "rebalance_days": days,
                "offset": offset,
                "max_rebalance_turnover_limit": args.max_rebalance_turnover,
                **{name: metrics.get(name) for name in METRICS},
                "summary_path": str(summary_path),
            })

    detail = pd.DataFrame(rows).sort_values(["rebalance_days", "offset"])
    summary_rows = []
    for days, group in detail.groupby("rebalance_days", sort=True):
        summary_rows.append({
            "rebalance_days": int(days),
            "n_offsets": int(len(group)),
            "net_ir_median": group["information_ratio"].median(),
            "net_ir_worst": group["information_ratio"].min(),
            "net_ir_positive_ratio": (group["information_ratio"] > 0).mean(),
            "annual_excess_median": group["annual_excess_return"].median(),
            "annual_excess_worst": group["annual_excess_return"].min(),
            "max_drawdown_median": group["max_drawdown"].median(),
            "max_drawdown_worst": group["max_drawdown"].min(),
            "daily_turnover_median": group["average_daily_turnover"].median(),
            "annualized_turnover_median": group["annualized_two_sided_turnover"].median(),
            "actual_exposure_median": group["average_actual_exposure"].median(),
            "total_cost_median": group["total_cost"].median(),
            "annualized_cost_ratio_median": group["annualized_cost_ratio"].median(),
        })
    summary = pd.DataFrame(summary_rows).sort_values("net_ir_median", ascending=False)

    out_dir = REPORTS_DIR / "rebalance_frequency" / args.model / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "offset_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "frequency_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))
    print(f"[OK] 汇总结果: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
