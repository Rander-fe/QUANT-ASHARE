"""Evaluate one QUANTMIND experimental factor without touching the formal library."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "reports/quantmind_trials/20260901T064435Z"
OUT = TRIAL / "evaluation"
FACTOR = "QM_DS_LIQUIDITY_AMPLIFICATION"
START, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2023-04-30")
MIN_CROSS_SECTION = 30
EPSILON = 1e-12


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = numerator.notna() & denominator.notna() & (denominator.abs() > EPSILON)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def calculate_exchange(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["symbol", "date", "close", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"] <= END].sort_values(["symbol", "date"], kind="mergesort")
    grouped = df.groupby("symbol", sort=False, observed=True)
    mean5 = grouped["amount"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    mean20 = grouped["amount"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    std20 = grouped["close"].rolling(20, min_periods=20).std(ddof=1).reset_index(level=0, drop=True)
    ratio = safe_div(mean20, mean5)
    df[FACTOR] = safe_div(mean5, std20 * ratio)
    return df.loc[(df["date"] >= START) & (df["date"] <= END), ["symbol", "date", FACTOR]]


def distribution(values: pd.Series) -> dict:
    finite = values[np.isfinite(values)].astype("float64")
    qs = finite.quantile([0, .0001, .001, .01, .05, .5, .95, .99, .999, .9999, 1])
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    extreme_mad = float(((finite - median).abs() > 10 * mad).mean()) if mad > 0 else None
    return {
        "count": int(finite.size), "mean": float(finite.mean()), "std": float(finite.std()),
        "skew": float(finite.skew()), "kurtosis": float(finite.kurt()), "median": median,
        "mad": mad, "outside_median_10mad_rate": extreme_mad,
        "quantiles": {str(k): float(v) for k, v in qs.items()},
    }


def correlations(group: pd.DataFrame) -> pd.Series:
    valid = group[[FACTOR, "label_ret_20"]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    if n < MIN_CROSS_SECTION or valid[FACTOR].nunique() < 2 or valid["label_ret_20"].nunique() < 2:
        return pd.Series({"n": n, "ic": np.nan, "rank_ic": np.nan})
    return pd.Series({
        "n": n,
        "ic": valid[FACTOR].corr(valid["label_ret_20"], method="pearson"),
        "rank_ic": valid[FACTOR].corr(valid["label_ret_20"], method="spearman"),
    })


def metric_summary(frame: pd.DataFrame) -> dict:
    result = {}
    for col in ("ic", "rank_ic"):
        x = frame[col].dropna()
        mean, std = float(x.mean()), float(x.std())
        result[col] = {
            "days": int(x.size), "mean": mean, "std": std,
            "ir": mean / std if std > 0 else None,
            "positive_rate": float((x > 0).mean()),
            "t_stat": mean / (std / np.sqrt(x.size)) if std > 0 and x.size else None,
            "direction_aligned_mean": -mean,
            "direction_aligned_positive_rate": float((x < 0).mean()),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = json.loads((TRIAL / "candidate.json").read_text(encoding="utf-8"))
    panel_path = OUT / "factor_values.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        factors = panel[["symbol", "date", FACTOR]]
        print(f"reused calculated panel: {len(panel):,}", flush=True)
    else:
        pieces = []
        for exchange in ("sh", "sz", "bj"):
            path = ROOT / f"data/raw/daily/daily_{exchange}.parquet"
            part = calculate_exchange(path)
            pieces.append(part)
            print(f"calculated {exchange}: {len(part):,}", flush=True)
        factors = pd.concat(pieces, ignore_index=True)
    total = len(factors)
    infinity_count = int(np.isinf(factors[FACTOR].to_numpy(dtype="float64", na_value=np.nan)).sum())
    missing_count = int(factors[FACTOR].isna().sum())

    if not panel_path.exists():
        labels = pd.read_parquet(ROOT / "data/processed/labels.parquet", columns=["symbol", "date", "label_ret_20"])
        labels["date"] = pd.to_datetime(labels["date"])
        labels = labels.loc[(labels["date"] >= START) & (labels["date"] <= END)]
        panel = factors.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")
        panel.to_parquet(panel_path, index=False)
    print(f"merged panel: {len(panel):,}; calculating daily IC", flush=True)

    daily = panel.groupby("date", sort=True, observed=True).apply(correlations, include_groups=False).reset_index()
    daily.to_parquet(OUT / "daily_ic.parquet", index=False)
    yearly_rows = []
    for year, group in daily.groupby(daily["date"].dt.year):
        row = {"year": int(year), **metric_summary(group)}
        yearly_rows.append({
            "year": row["year"], "calendar_trading_days": int(len(group)),
            "valid_ic_days": row["ic"]["days"], "valid_rank_ic_days": row["rank_ic"]["days"],
            "ic_mean": row["ic"]["mean"], "ic_ir": row["ic"]["ir"],
            "ic_positive_rate": row["ic"]["positive_rate"],
            "rank_ic_mean": row["rank_ic"]["mean"], "rank_ic_ir": row["rank_ic"]["ir"],
            "rank_ic_positive_rate": row["rank_ic"]["positive_rate"],
            "aligned_ic_mean": row["ic"]["direction_aligned_mean"],
            "aligned_rank_ic_mean": row["rank_ic"]["direction_aligned_mean"],
        })
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUT / "yearly_stability.csv", index=False, encoding="utf-8-sig")

    report = {
        "status": "experimental_only_not_sota",
        "factor_name": FACTOR,
        "formula": candidate.get("formula") or candidate["candidate"]["formula"],
        "declared_direction": "negative",
        "protocol": {"period": [str(START.date()), str(END.date())], "label": "label_ret_20",
                     "daily_cross_section_min_n": MIN_CROSS_SECTION, "validation_or_test_used": False},
        "quality": {"rows": total, "missing_count": missing_count,
                    "missing_rate": missing_count / total, "infinity_count": infinity_count},
        "extreme_values": distribution(factors[FACTOR]),
        "daily_performance": metric_summary(daily),
        "yearly_stability": yearly_rows,
        "notes": [
            "Raw IC is reported; direction-aligned metrics multiply the declared negative-direction factor by -1.",
            "The formula uses absolute close-price volatility, so it may contain price-level and amount-size exposure.",
            "No winsorization, neutralization, or standardization was applied in this diagnostic run.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["daily_performance"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
