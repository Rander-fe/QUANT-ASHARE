"""PIT industry-relative and industry-residual return atoms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_industry_intervals(daily: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Attach membership with half-open interval semantics: start_date <= date < end_date."""
    intervals = intervals.copy()
    daily = daily.copy()
    daily["symbol"] = daily["symbol"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"])
    intervals["symbol"] = intervals["symbol"].astype(str)
    intervals["start_date"] = pd.to_datetime(intervals["start_date"])
    intervals["end_date"] = pd.to_datetime(intervals["end_date"])
    merged = pd.merge_asof(
        daily.sort_values(["date", "symbol"]),
        intervals[["symbol", "start_date", "end_date", "industry_code", "industry_name"]]
        .sort_values(["start_date", "symbol"]),
        left_on="date", right_on="start_date", by="symbol", direction="backward"
    )
    active = merged["end_date"].isna() | merged["date"].lt(merged["end_date"])
    merged.loc[~active, ["industry_code", "industry_name"]] = pd.NA
    return merged.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_industry_atoms(
    frame: pd.DataFrame, relative_window: int = 20, regression_window: int = 60,
    residual_momentum_window: int = 20, min_industry_members: int = 5,
) -> pd.DataFrame:
    """Build leave-one-out industry relative return and market+industry regression residual momentum."""
    required = {"symbol", "date", "close", "industry_code"}
    if not required <= set(frame):
        raise KeyError(f"缺少字段: {sorted(required-set(frame))}")
    x = frame.sort_values(["symbol", "date"]).copy()
    x["stock_ret_1d"] = x.groupby("symbol", observed=True)["close"].pct_change(fill_method=None)
    grouped = x.groupby(["date", "industry_code"], observed=True)["stock_ret_1d"]
    count = grouped.transform("count")
    total = grouped.transform("sum")
    x["industry_ret_1d_loo"] = ((total - x["stock_ret_1d"]) / (count - 1)).where(count >= min_industry_members)
    market = x.groupby("date", observed=True)["stock_ret_1d"].mean()
    x["market_ret_1d"] = x["date"].map(market)

    x["stock_ret_20"] = x.groupby("symbol", observed=True)["close"].pct_change(relative_window, fill_method=None)
    industry_growth = (1.0 + x["industry_ret_1d_loo"]).groupby(x["symbol"], observed=True).rolling(
        relative_window, min_periods=relative_window
    ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1.0
    x["IND_REL_RET_20"] = x["stock_ret_20"] - industry_growth

    def roll_mean(values: pd.Series) -> pd.Series:
        return values.groupby(x["symbol"], observed=True).rolling(
            regression_window, min_periods=regression_window
        ).mean().reset_index(level=0, drop=True)

    y, i, m = x["stock_ret_1d"], x["industry_ret_1d_loo"], x["market_ret_1d"]
    my, mi, mm = roll_mean(y), roll_mean(i), roll_mean(m)
    var_i, var_m = roll_mean(i*i) - mi*mi, roll_mean(m*m) - mm*mm
    cov_im = roll_mean(i*m) - mi*mm
    cov_yi, cov_ym = roll_mean(y*i) - my*mi, roll_mean(y*m) - my*mm
    determinant = var_i*var_m - cov_im*cov_im
    valid = determinant.abs() > 1e-18
    beta_i = ((cov_yi*var_m - cov_ym*cov_im) / determinant).where(valid)
    beta_m = ((cov_ym*var_i - cov_yi*cov_im) / determinant).where(valid)
    alpha = my - beta_i*mi - beta_m*mm
    x["IND_RESID_RET_1D"] = y - alpha - beta_i*i - beta_m*m
    x["IND_RESID_MOM_20"] = x.groupby("symbol", observed=True)["IND_RESID_RET_1D"].rolling(
        residual_momentum_window, min_periods=residual_momentum_window
    ).sum().reset_index(level=0, drop=True)
    return x
