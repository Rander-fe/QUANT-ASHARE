"""Compare the original QUANTMIND liquidity factor with three safer variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "reports/quantmind_trials/20260901T064435Z"
BASE_EVAL = TRIAL / "evaluation"
OUT = TRIAL / "variant_comparison"
START, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2023-04-30")
EPS = 1e-12
MIN_N = 30
ORIGINAL = "QM_DS_LIQUIDITY_AMPLIFICATION"
WINSOR_Z = f"{ORIGINAL}_WINSOR_Z"
RETVOL = f"{ORIGINAL}_RETVOL"
TURNOVER = f"{ORIGINAL}_TURNOVER_RETVOL"
FACTORS = [ORIGINAL, WINSOR_Z, RETVOL, TURNOVER]


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    valid = a.notna() & b.notna() & (b.abs() > EPS)
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out


def rolling_factor(frame: pd.DataFrame, input_col: str, volatility: pd.Series) -> pd.Series:
    grouped_input = frame.groupby("symbol", sort=False, observed=True)[input_col]
    short = grouped_input.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    long = grouped_input.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    return safe_div(short, volatility * safe_div(long, short))


def winsor_z(group: pd.Series) -> pd.Series:
    valid = group.replace([np.inf, -np.inf], np.nan)
    lo, hi = valid.quantile([0.01, 0.99])
    clipped = valid.clip(lo, hi)
    std = clipped.std(ddof=1)
    return (clipped - clipped.mean()) / std if pd.notna(std) and std > EPS else clipped * np.nan


def daily_metrics(day: pd.DataFrame, factor: str) -> dict:
    cols = [factor, "label_ret_20", "circ_mv", "industry"]
    work = day[cols].replace([np.inf, -np.inf], np.nan)
    valid_ic = work[[factor, "label_ret_20"]].dropna()
    result = {"n": len(valid_ic), "ic": np.nan, "rank_ic": np.nan,
              "size_rank_corr": np.nan, "industry_eta2": np.nan}
    if len(valid_ic) >= MIN_N and valid_ic[factor].nunique() > 1:
        result["ic"] = valid_ic[factor].corr(valid_ic["label_ret_20"], method="pearson")
        result["rank_ic"] = valid_ic[factor].corr(valid_ic["label_ret_20"], method="spearman")
    valid_size = work.loc[(work["circ_mv"] > 0), [factor, "circ_mv"]].dropna()
    if len(valid_size) >= MIN_N and valid_size[factor].nunique() > 1:
        result["size_rank_corr"] = valid_size[factor].corr(np.log(valid_size["circ_mv"]), method="spearman")
    valid_ind = work[[factor, "industry"]].dropna()
    if len(valid_ind) >= MIN_N and valid_ind[factor].nunique() > 1:
        ranks = valid_ind[factor].rank(pct=True)
        overall = ranks.mean()
        grouped = pd.DataFrame({"x": ranks, "industry": valid_ind["industry"]}).groupby("industry")["x"]
        counts, means = grouped.size(), grouped.mean()
        between = float((counts * (means - overall) ** 2).sum())
        total = float(((ranks - overall) ** 2).sum())
        result["industry_eta2"] = between / total if total > EPS else np.nan
    return result


def summarize(daily: pd.DataFrame, factor: str) -> dict:
    block = daily[daily["factor"] == factor]
    result = {"factor": factor, "days": int(block["ic"].notna().sum())}
    for col in ("ic", "rank_ic", "size_rank_corr", "industry_eta2"):
        x = block[col].dropna()
        result[f"{col}_mean"] = float(x.mean())
        result[f"{col}_std"] = float(x.std())
    result["aligned_ic_mean"] = -result["ic_mean"]
    result["aligned_rank_ic_mean"] = -result["rank_ic_mean"]
    result["aligned_ic_ir"] = -result["ic_mean"] / result["ic_std"]
    result["aligned_rank_ic_ir"] = -result["rank_ic_mean"] / result["rank_ic_std"]
    result["aligned_ic_correct_rate"] = float((block["ic"].dropna() < 0).mean())
    result["aligned_rank_ic_correct_rate"] = float((block["rank_ic"].dropna() < 0).mean())
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_EVAL / "factor_values.parquet")
    base["date"] = pd.to_datetime(base["date"])
    extra_cols = ["symbol", "date", "close", "turnover_rate", "circ_mv", "industry"]
    extra = pd.read_parquet(
        ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
        columns=extra_cols,
        filters=[[('date', '>=', START), ('date', '<=', END)]],
    )
    extra["date"] = pd.to_datetime(extra["date"])
    for col in ("turnover_rate", "circ_mv"):
        extra[col] = pd.to_numeric(extra[col], errors="coerce").mask(lambda x: x == 0)
    frame = base.merge(extra, on=["symbol", "date"], how="left", validate="one_to_one")
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    returns = frame.groupby("symbol", sort=False, observed=True)["close"].pct_change(fill_method=None)
    retvol = returns.groupby(frame["symbol"], sort=False).rolling(20, min_periods=20).std(ddof=1).reset_index(level=0, drop=True)

    # Reconstruct amount-based return-volatility variant from the raw daily exchange files.
    amounts = []
    for exchange in ("sh", "sz", "bj"):
        part = pd.read_parquet(ROOT / f"data/raw/daily/daily_{exchange}.parquet", columns=["symbol", "date", "amount"])
        part["date"] = pd.to_datetime(part["date"])
        amounts.append(part.loc[(part["date"] >= START) & (part["date"] <= END)])
    amount = pd.concat(amounts, ignore_index=True)
    frame = frame.merge(amount, on=["symbol", "date"], how="left", validate="one_to_one")
    frame[RETVOL] = rolling_factor(frame, "amount", retvol)
    frame[TURNOVER] = rolling_factor(frame, "turnover_rate", retvol)
    frame[WINSOR_Z] = frame.groupby("date", sort=False, observed=True)[ORIGINAL].transform(winsor_z)
    frame = frame.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)

    daily_rows = []
    for date, day in frame.groupby("date", sort=True, observed=True):
        for factor in FACTORS:
            daily_rows.append({"date": date, "factor": factor, **daily_metrics(day, factor)})
    daily = pd.DataFrame(daily_rows)
    daily.to_parquet(OUT / "daily_comparison.parquet", index=False)
    summary = pd.DataFrame([summarize(daily, factor) for factor in FACTORS])
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    yearly_rows = []
    daily["year"] = daily["date"].dt.year
    for (factor, year), block in daily.groupby(["factor", "year"], sort=True):
        yearly_rows.append({"factor": factor, "year": int(year),
                            "aligned_ic_mean": -float(block["ic"].mean()),
                            "aligned_rank_ic_mean": -float(block["rank_ic"].mean()),
                            "size_rank_corr_mean": float(block["size_rank_corr"].mean()),
                            "industry_eta2_mean": float(block["industry_eta2"].mean())})
    pd.DataFrame(yearly_rows).to_csv(OUT / "yearly_comparison.csv", index=False, encoding="utf-8-sig")

    quality = []
    for factor in FACTORS:
        x = frame[factor]
        finite = x[np.isfinite(x)]
        quality.append({"factor": factor, "rows": len(x), "missing_rate": float(x.isna().mean()),
                        "infinity_count": int(np.isinf(x.to_numpy(dtype=float, na_value=np.nan)).sum()),
                        "p01": float(finite.quantile(.01)), "median": float(finite.median()),
                        "p99": float(finite.quantile(.99)), "max": float(finite.max()),
                        "skew": float(finite.skew()), "kurtosis": float(finite.kurt())})
    pd.DataFrame(quality).to_csv(OUT / "quality_extremes.csv", index=False, encoding="utf-8-sig")
    frame[["symbol", "date", *FACTORS, "label_ret_20", "circ_mv", "industry"]].to_parquet(
        OUT / "variant_factor_values.parquet", index=False)

    report = {
        "status": "experimental_comparison_not_sota", "period": [str(START.date()), str(END.date())],
        "label": "label_ret_20", "direction": "negative", "test_period_used": False,
        "variants": {
            ORIGINAL: "original",
            WINSOR_Z: "daily cross-sectional 1%/99% winsorization followed by z-score",
            RETVOL: "replace 20-day close-level std with 20-day close-return std",
            TURNOVER: "replace amount with Tushare turnover_rate and use 20-day return std",
        },
        "industry_warning": "industry is a current stock_basic snapshot backfilled historically; exposure is diagnostic only",
        "summary": summary.to_dict(orient="records"), "quality": quality,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
