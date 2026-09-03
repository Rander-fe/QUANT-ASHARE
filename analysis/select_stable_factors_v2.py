"""按训练期多阶段RankIC稳定性选择V2因子，不读取验证/测试收益。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, TRAIN_PERIOD


def _stats(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return pd.DataFrame({
        f"ic_{suffix}": frame.mean(),
        f"std_{suffix}": frame.std(ddof=1),
        f"positive_{suffix}": (frame > 0).mean(),
        f"count_{suffix}": frame.count(),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="V2因子衰减诊断与稳定性筛选")
    parser.add_argument("--input", default="v2_label20_daily_rankic.parquet")
    parser.add_argument("--max-factors", type=int, default=40)
    parser.add_argument("--corr-threshold", type=float, default=0.70)
    args = parser.parse_args()

    path = DATA_PROCESSED / args.input
    daily = pd.read_parquet(path)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.loc[pd.Timestamp(TRAIN_PERIOD[0]):pd.Timestamp(TRAIN_PERIOD[1])]
    if daily.empty:
        raise ValueError(f"训练期日频IC为空: {path}")

    # 三段均完全位于训练集；近期段用于识别衰减，不触碰验证/测试期。
    early = daily.loc[:"2019-12-31"]
    middle = daily.loc["2020-01-01":"2021-12-31"]
    recent = daily.loc["2022-01-01":]
    report = _stats(daily, "full")
    report = report.join(_stats(early, "early")).join(_stats(middle, "middle"))
    report = report.join(_stats(recent, "recent"))

    # 方向只由较早训练段确定；后续统计全部按该方向对齐。
    direction = np.sign(report["ic_early"]).replace(0, np.nan)
    report["direction"] = direction
    for period in ("full", "early", "middle", "recent"):
        report[f"aligned_ic_{period}"] = report[f"ic_{period}"] * direction
    report["recent_retention"] = (
        report["aligned_ic_recent"] / report["aligned_ic_early"].abs().clip(lower=1e-4)
    )
    report["same_sign_periods"] = (
        (report["aligned_ic_middle"] > 0).astype(int)
        + (report["aligned_ic_recent"] > 0).astype(int)
    )
    report["worst_period_ic"] = report[
        ["aligned_ic_early", "aligned_ic_middle", "aligned_ic_recent"]
    ].min(axis=1)
    report["stability_score"] = (
        0.40 * report["aligned_ic_recent"]
        + 0.25 * report["aligned_ic_full"]
        + 0.20 * report["worst_period_ic"]
        + 0.15 * report["positive_recent"].sub(0.5)
    )
    report.index.name = "factor"
    report = report.reset_index().sort_values("stability_score", ascending=False)

    eligible = report[
        (report["count_recent"] >= 120)
        & (report["aligned_ic_full"] >= 0.008)
        & (report["aligned_ic_recent"] >= 0.005)
        & (report["same_sign_periods"] == 2)
        & (report["recent_retention"] >= 0.30)
    ].copy()

    # 基于日频IC序列贪心去冗余，优先保留近期稳定性得分更高者。
    kept: list[str] = []
    aligned = daily.mul(direction, axis=1)
    for factor in eligible["factor"]:
        if len(kept) >= args.max_factors:
            break
        if kept:
            corr = aligned[kept].corrwith(aligned[factor]).abs()
            if bool((corr >= args.corr_threshold).any()):
                continue
        kept.append(factor)
    report["selected_v2"] = report["factor"].isin(kept)

    out_dir = DATA_PROCESSED.parent.parent / "reports" / "factor_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "factor_decay_report.csv", index=False)
    payload = {
        "version": "v2_train_only_stability",
        "label": "label_ret_20",
        "source": str(path),
        "periods": {"early": ["2016-01-01", "2019-12-31"],
                    "middle": ["2020-01-01", "2021-12-31"],
                    "recent": ["2022-01-01", "2023-04-30"]},
        "selection": {"max_factors": args.max_factors,
                      "corr_threshold": args.corr_threshold,
                      "min_full_aligned_ic": 0.008,
                      "min_recent_aligned_ic": 0.005,
                      "min_recent_retention": 0.30,
                      "same_sign_middle_recent": True},
        "n_selected": len(kept), "factors": kept,
        "test_data_used": False,
    }
    output = DATA_PROCESSED / "passed_factor_cols_v2.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] V2稳定因子: {len(kept)} -> {output}")
    print(report.head(30)[["factor", "ic_early", "ic_recent", "recent_retention",
                           "stability_score", "selected_v2"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
