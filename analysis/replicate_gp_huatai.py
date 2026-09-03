# -*- coding: utf-8 -*-
"""复现并审计华泰遗传规划报告的六个选股因子。

仅使用训练集和验证集；不读取 TEST_PERIOD。输出原始因子、中性化因子、
分段 RankIC、Newey-West/非重叠统计以及与现有因子日频 IC 的冗余关系。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BASIC_EXTRA_PATH, DATA_PROCESSED, REPORTS_DIR, TRAIN_PERIOD, VALID_PERIOD
from factors.base.gp_huatai_replication import FACTOR_NAMES
from factors.registry import get_factor
from models.lightgbm.evaluate import calc_ic, evaluate_predictions
from research.protocol import write_experiment_manifest

warnings.filterwarnings("ignore", message=r".*input array is constant.*")
EPS = 1e-12
OUTPUT_DIR = REPORTS_DIR / "gp_huatai_replication"
RAW_OUTPUT = DATA_PROCESSED / "gp_huatai_raw.parquet"
NEUTRAL_OUTPUT = DATA_PROCESSED / "gp_huatai_neutralized.parquet"


def _load_source(max_rows: int | None = None) -> pd.DataFrame:
    columns = ["symbol", "date", "high", "low", "close", "volume", "amount",
               "turnover_rate", "industry", "total_mv"]
    frame = pd.read_parquet(BASIC_EXTRA_PATH, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    # 明确截断到验证期末，杜绝本复现流程接触最终测试期。
    frame = frame[frame["date"] <= pd.Timestamp(VALID_PERIOD[1])]
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    if max_rows:
        frame = frame.iloc[:max_rows].copy()
    return frame


def _build_controls(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("symbol", sort=False)
    returns = grouped["close"].pct_change(fill_method=None)
    frame["style_ret20"] = grouped["close"].transform(
        lambda s: s / s.shift(20) - 1
    )
    frame["style_turn20"] = grouped["turnover_rate"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    frame["style_vol20"] = returns.groupby(frame["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    mv = pd.to_numeric(frame["total_mv"], errors="coerce") * 1e4
    frame["style_log_mv"] = np.log(mv.clip(lower=1.0))
    return frame


def _neutralize_day(day: pd.DataFrame) -> pd.DataFrame:
    out = day[["symbol", "date", *FACTOR_NAMES]].copy()
    # OLS residuals are float64.  Promote the destination once so pandas does
    # not repeatedly warn (and copy blocks) for every factor on every day.
    factor_cols = list(FACTOR_NAMES)
    out[factor_cols] = out[factor_cols].astype(np.float64)
    controls = ["style_log_mv", "style_ret20", "style_turn20", "style_vol20"]
    valid = day["industry"].notna() & day[controls].notna().all(axis=1)
    if valid.sum() < 50:
        out.loc[:, FACTOR_NAMES] = np.nan
        return out
    source = day.loc[valid]
    industries = pd.get_dummies(source["industry"], dtype=float, drop_first=False)
    numeric = source[controls].astype(float)
    numeric = (numeric - numeric.mean()) / (numeric.std(ddof=1) + EPS)
    design = np.column_stack([industries.to_numpy(), numeric.to_numpy()])
    for factor in FACTOR_NAMES:
        y = pd.to_numeric(source[factor], errors="coerce")
        mask = y.notna() & np.isfinite(y)
        if mask.sum() < max(50, design.shape[1] + 5):
            continue
        values = y.loc[mask].to_numpy(dtype=float)
        # MAD 去极值，与报告的中位数±5*MAD口径一致。
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        if np.isfinite(mad) and mad > EPS:
            values = np.clip(values, median - 5 * mad, median + 5 * mad)
        x = design[mask.to_numpy()]
        beta, _, _, _ = np.linalg.lstsq(x, values, rcond=None)
        residual = values - x @ beta
        residual = (residual - residual.mean()) / (residual.std(ddof=1) + EPS)
        out.loc[source.index[mask], factor] = residual
    return out


def build_factors(max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _build_controls(_load_source(max_rows=max_rows))
    for name in FACTOR_NAMES:
        frame[name] = get_factor(name)["func"](frame).astype(np.float32)
        print(f"[OK] 已计算 {name}")
    labels = frame.groupby("symbol", sort=False)["close"].transform(
        lambda s: s.shift(-20) / s - 1
    )
    raw = frame[["symbol", "date", *FACTOR_NAMES]].copy()
    raw["label_ret_20"] = labels.astype(np.float32)
    raw.to_parquet(RAW_OUTPUT, index=False)

    parts = []
    for i, (_, day) in enumerate(frame.groupby("date", sort=True), start=1):
        parts.append(_neutralize_day(day))
        if i % 250 == 0:
            print(f"[INFO] 已中性化 {i} 个交易日")
    neutral = pd.concat(parts, ignore_index=True)
    neutral[list(FACTOR_NAMES)] = neutral[list(FACTOR_NAMES)].astype(np.float32)
    neutral = neutral.merge(raw[["symbol", "date", "label_ret_20"]],
                            on=["symbol", "date"], how="left", validate="one_to_one")
    neutral.to_parquet(NEUTRAL_OUTPUT, index=False)
    return raw, neutral


def _segment_metrics(frame: pd.DataFrame, value_col: str, factor: str,
                     segment: str, period: tuple[str, str]) -> dict:
    part = frame[(frame["date"] >= pd.Timestamp(period[0])) &
                 (frame["date"] <= pd.Timestamp(period[1]))]
    # 训练期末20日标签跨入验证期，必须purge。
    if segment == "train":
        dates = np.sort(part["date"].unique())
        if len(dates) > 20:
            part = part[part["date"] <= dates[-21]]
    metrics = evaluate_predictions(
        part.rename(columns={factor: value_col}), value_col, "label_ret_20"
    )
    return {"factor": factor, "variant": frame.attrs.get("variant"),
            "segment": segment, **metrics}


def audit(raw: pd.DataFrame, neutral: pd.DataFrame) -> pd.DataFrame:
    raw.attrs["variant"] = "raw"
    neutral.attrs["variant"] = "industry_plus_4_styles"
    rows = []
    for frame in (raw, neutral):
        for factor in FACTOR_NAMES:
            rows.append(_segment_metrics(frame, "pred", factor, "train", TRAIN_PERIOD))
            rows.append(_segment_metrics(frame, "pred", factor, "valid", VALID_PERIOD))
    report = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "rankic_summary.csv", index=False, encoding="utf-8-sig")

    # 用日频 RankIC 序列相关性检查与现有因子预测行为是否重复。
    train = neutral[(neutral["date"] >= pd.Timestamp(TRAIN_PERIOD[0])) &
                    (neutral["date"] <= pd.Timestamp(TRAIN_PERIOD[1]))]
    gp_daily = {}
    for factor in FACTOR_NAMES:
        gp_daily[factor] = calc_ic(train, factor, "label_ret_20")["rank_ic"]
    gp_daily = pd.DataFrame(gp_daily)
    existing_path = DATA_PROCESSED / "v2_label20_daily_rankic.parquet"
    redundancy_rows = []
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing.index = pd.to_datetime(existing.index)
        for factor in FACTOR_NAMES:
            correlations = existing.corrwith(gp_daily[factor]).dropna().abs().sort_values(ascending=False)
            if len(correlations):
                redundancy_rows.append({"factor": factor,
                                        "most_similar": correlations.index[0],
                                        "daily_ic_corr_abs": correlations.iloc[0],
                                        "redundant_at_0_7": correlations.iloc[0] >= 0.7})
    pd.DataFrame(redundancy_rows).to_csv(
        OUTPUT_DIR / "redundancy.csv", index=False, encoding="utf-8-sig"
    )
    write_experiment_manifest(
        OUTPUT_DIR / "manifest.json", experiment="huatai_gp_six_factor_replication",
        config={"discovery_source": "HTSC 2019-06-10", "label": "label_ret_20",
                "train_period": list(TRAIN_PERIOD), "valid_period": list(VALID_PERIOD),
                "neutralization": ["industry", "log_mv", "ret20", "turn20", "vol20"],
                "test_data_used": False},
        inputs=[BASIC_EXTRA_PATH], features=list(FACTOR_NAMES), test_data_used=False,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="复现华泰遗传规划六因子")
    parser.add_argument("--max-rows", type=int, help="仅用于冒烟测试")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        raw = pd.read_parquet(RAW_OUTPUT)
        neutral = pd.read_parquet(NEUTRAL_OUTPUT)
    else:
        raw, neutral = build_factors(max_rows=args.max_rows)
    report = audit(raw, neutral)
    print(report[["factor", "variant", "segment", "rank_ic_mean", "rank_ic_nw_t",
                  "rank_ic_nonoverlap_median", "long_short_spread"]].to_string(index=False))
    print(f"[DONE] 报告: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
