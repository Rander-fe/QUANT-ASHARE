"""验证4个反转/动量修正因子；绝不覆盖正式 factors.parquet。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import BASIC_EXTRA_PATH, DATA_PROCESSED, REPORTS_DIR, TRAIN_PERIOD
from factors.base.reversal_momentum import calc_consec_up, calc_max_drawdown

NEW_COLS = ["CONSEC_UP3_V2", "CONSEC_UP5_V2", "MAX_DD20_V2", "MAX_DD60_V2"]
OLD_TO_NEW = {
    "CONSEC_UP3": "CONSEC_UP3_V2", "CONSEC_UP5": "CONSEC_UP5_V2",
    "MAX_DD20": "MAX_DD20_V2", "MAX_DD60": "MAX_DD60_V2",
}
RANGES = {
    "CONSEC_UP3_V2": (-3.0, 0.0), "CONSEC_UP5_V2": (-5.0, 0.0),
    "MAX_DD20_V2": (0.0, 1.0), "MAX_DD60_V2": (0.0, 1.0),
}


def build_v2(output: Path) -> pd.DataFrame:
    base = pd.read_parquet(BASIC_EXTRA_PATH, columns=["symbol", "date", "close"])
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["symbol", "date"]).reset_index(drop=True)
    result = base[["symbol", "date"]].copy()
    result["CONSEC_UP3_V2"] = calc_consec_up(base, 3).astype("float32")
    result["CONSEC_UP5_V2"] = calc_consec_up(base, 5).astype("float32")
    grouped = base.groupby("symbol", sort=False)["close"]
    result["MAX_DD20_V2"] = grouped.transform(lambda s: calc_max_drawdown(s, 20)).astype("float32")
    result["MAX_DD60_V2"] = grouped.transform(lambda s: calc_max_drawdown(s, 60)).astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def _daily_correlations(frame: pd.DataFrame, factor: str, label: str) -> pd.DataFrame:
    rows = []
    for date, day in frame[["date", factor, label]].groupby("date", sort=True):
        valid = day[[factor, label]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 30 or valid[factor].nunique() < 2 or valid[label].nunique() < 2:
            continue
        rows.append({"date": date, "ic": valid[factor].corr(valid[label]),
                     "rank_ic": valid[factor].corr(valid[label], method="spearman")})
    return pd.DataFrame(rows)


def _ic_summary(frame: pd.DataFrame, factor: str, label: str, segment: str) -> dict:
    daily = _daily_correlations(frame, factor, label)
    if daily.empty:
        return {"factor": factor, "segment": segment, "days": 0}
    return {
        "factor": factor, "segment": segment, "days": int(len(daily)),
        "ic_mean": float(daily["ic"].mean()), "rank_ic_mean": float(daily["rank_ic"].mean()),
        "rank_ic_std": float(daily["rank_ic"].std()),
        "rank_ic_positive_ratio": float((daily["rank_ic"] > 0).mean()),
    }


def _factor_quality(frame: pd.DataFrame, factor: str) -> dict:
    values = pd.to_numeric(frame[factor], errors="coerce")
    finite = np.isfinite(values)
    lo, hi = RANGES[factor]
    finite_values = values[finite]
    return {
        "factor": factor, "rows": int(len(values)),
        "missing_rate": float(values.isna().mean()), "infinite_count": int(np.isinf(values).sum()),
        "min": float(finite_values.min()), "q001": float(finite_values.quantile(.001)),
        "q01": float(finite_values.quantile(.01)), "median": float(finite_values.median()),
        "q99": float(finite_values.quantile(.99)), "q999": float(finite_values.quantile(.999)),
        "max": float(finite_values.max()), "expected_min": lo, "expected_max": hi,
        "range_violation_count": int(((finite_values < lo - 1e-7) | (finite_values > hi + 1e-7)).sum()),
    }


def validate(v2_path: Path, label: str) -> dict:
    start, end = TRAIN_PERIOD
    filters = [("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))]
    v2 = pd.read_parquet(v2_path, filters=filters)
    old = pd.read_parquet(DATA_PROCESSED / "factors.parquet",
                          columns=["symbol", "date", label] + list(OLD_TO_NEW), filters=filters)
    meta = pd.read_parquet(BASIC_EXTRA_PATH,
                           columns=["symbol", "date", "industry", "total_mv"], filters=filters)
    for item in (v2, old, meta): item["date"] = pd.to_datetime(item["date"])
    frame = old.merge(v2, on=["symbol", "date"], how="inner", validate="one_to_one")
    frame = frame.merge(meta, on=["symbol", "date"], how="left", validate="one_to_one")
    symbols = frame["symbol"].astype(str)
    prefix_exchange = symbols.str.extract(r"^(SH|SZ|BJ)", expand=False)
    suffix_exchange = symbols.str.extract(r"\.(SH|SZ|BJ)$", expand=False)
    frame["exchange"] = prefix_exchange.fillna(suffix_exchange).fillna("UNKNOWN")
    frame["year"] = frame["date"].dt.year
    frame["size_group"] = frame.groupby("date")["total_mv"].transform(
        lambda s: pd.qcut(s.where(s > 0), 3, labels=["small", "mid", "large"], duplicates="drop")
    )

    quality = [_factor_quality(frame, factor) for factor in NEW_COLS]
    comparisons = []
    for old_name, new_name in OLD_TO_NEW.items():
        pair = frame[[old_name, new_name]].replace([np.inf, -np.inf], np.nan).dropna()
        comparisons.append({
            "old_factor": old_name, "new_factor": new_name, "paired_rows": int(len(pair)),
            "pearson": float(pair[old_name].corr(pair[new_name])),
            "spearman": float(pair[old_name].corr(pair[new_name], method="spearman")),
            "mean_abs_change": float((pair[old_name] - pair[new_name]).abs().mean()),
        })

    overall, yearly, pools, industries = [], [], [], []
    # 行业字段本身有时点风险，只选训练期样本最多的10个行业做代表性诊断，
    # 避免对全部行业反复扫描数百万行；正式行业验证应等待PIT行业数据。
    diagnostic_industries = set(frame["industry"].value_counts().head(10).index)
    for factor in NEW_COLS:
        overall.append(_ic_summary(frame, factor, label, "train_all"))
        for year, part in frame.groupby("year"):
            yearly.append(_ic_summary(part, factor, label, str(year)))
        for exchange, part in frame.groupby("exchange"):
            pools.append(_ic_summary(part, factor, label, f"exchange:{exchange}"))
        for size, part in frame.dropna(subset=["size_group"]).groupby("size_group", observed=True):
            pools.append(_ic_summary(part, factor, label, f"size:{size}"))
        # 行业是当前快照回填，仅列覆盖最大的行业作风险诊断，不参与通过判断。
        industry_frame = frame[frame["industry"].isin(diagnostic_industries)]
        for industry, part in industry_frame.groupby("industry"):
            if part["date"].nunique() >= 100 and len(part) >= 5000:
                industries.append(_ic_summary(part, factor, label, f"industry:{industry}"))

    decisions = []
    for factor in NEW_COLS:
        q = next(x for x in quality if x["factor"] == factor)
        ic = next(x for x in overall if x["factor"] == factor)
        technical_pass = q["infinite_count"] == 0 and q["range_violation_count"] == 0 and ic.get("days", 0) >= 500
        decisions.append({"factor": factor, "technical_pass": technical_pass,
                          "research_pass": None,
                          "note": "技术检查通过；是否正式替换需人工查看年度/股票池稳定性，不使用测试期选择。" if technical_pass else "技术检查未通过。"})

    return {
        "protocol": {"label": label, "selection_period": list(TRAIN_PERIOD),
                     "test_period_used": False,
                     "industry_warning": "industry为当前stock_basic快照回填历史，只作诊断，不作为通过条件",
                     "official_factor_file_overwritten": False},
        "quality": quality, "old_new_comparison": comparisons,
        "overall_ic": overall, "yearly_ic": yearly, "pool_ic": pools,
        "industry_diagnostic": industries, "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="label_ret_20", choices=["label_ret_5", "label_ret_10", "label_ret_20"])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    v2_path = DATA_PROCESSED / "reversal_momentum_fixes_v2.parquet"
    if args.rebuild or not v2_path.exists(): build_v2(v2_path)
    report = validate(v2_path, args.label)
    out_dir = REPORTS_DIR / "factor_validation" / "reversal_momentum_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ["quality", "old_new_comparison", "overall_ic", "yearly_ic", "pool_ic", "industry_diagnostic", "decisions"]:
        pd.DataFrame(report[key]).to_csv(out_dir / f"{key}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"v2_file": str(v2_path), "report_dir": str(out_dir),
                      "decisions": report["decisions"]}, ensure_ascii=False))
    return 0 if all(x["technical_pass"] for x in report["decisions"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
