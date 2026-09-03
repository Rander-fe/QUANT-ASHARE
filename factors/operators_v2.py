"""QUANTMIND 可调用的第1层安全算子；旧算子保留用于历史复现。"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pandas as pd


def _window(window: int) -> int:
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window 必须是正整数")
    return window


def ts_delay(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(_window(periods))


def ts_delta(series: pd.Series, periods: int = 1) -> pd.Series:
    periods = _window(periods)
    return series - series.shift(periods)


def fin_lag_report(series: pd.Series, report_period: pd.Series, periods: int = 1) -> pd.Series:
    """按不同财报期滞后，而不是按交易日滞后。

    输入必须是单只股票、按日期排序的 PIT 日频快照。同一报告期修订不增加期数。
    """
    periods = _window(periods)
    if not series.index.equals(report_period.index):
        raise ValueError("FIN_LAG_REPORT 的值与report_period索引必须一致")
    periods_dt = pd.to_datetime(report_period, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    known: dict[pd.Timestamp, float] = {}
    disclosed_periods: set[pd.Timestamp] = set()
    for idx, period, value in zip(series.index, periods_dt, pd.to_numeric(series, errors="coerce")):
        if pd.isna(period):
            continue
        disclosed_periods.add(period)
        if pd.notna(value):
            known[period] = float(value)
        ordered = sorted(disclosed_periods)
        position = ordered.index(period)
        if position >= periods:
            result.loc[idx] = known.get(ordered[position - periods], np.nan)
    return result


def fin_delta_report(series: pd.Series, report_period: pd.Series, periods: int = 1) -> pd.Series:
    """当前 PIT 财务值减去前 N 个不同财报期的值。"""
    return series - fin_lag_report(series, report_period, periods)


def safe_div(numerator, denominator, epsilon: float = 1e-12):
    if not isinstance(numerator, pd.Series) and not isinstance(denominator, pd.Series):
        if pd.isna(numerator) or pd.isna(denominator) or abs(denominator) <= epsilon:
            return np.nan
        return numerator / denominator
    if isinstance(numerator, pd.Series):
        index = numerator.index
    else:
        index = denominator.index
    num = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator, index=index)
    den = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=index)
    if not num.index.equals(den.index):
        raise ValueError("SAFE_DIV 两个序列的索引必须完全一致")
    valid = num.notna() & den.notna() & (den.abs() > epsilon)
    return (num / den).where(valid)


def abs_values(series: pd.Series) -> pd.Series:
    return series.abs()


def sign_values(series: pd.Series) -> pd.Series:
    return np.sign(series)


def log1p_safe(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = numeric.notna() & (numeric > -1.0)
    result.loc[valid] = np.log1p(numeric.loc[valid])
    return result


def sqrt_safe(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = numeric.notna() & (numeric >= 0.0)
    result.loc[valid] = np.sqrt(numeric.loc[valid])
    return result


def where_values(condition: pd.Series, when_true, when_false) -> pd.Series:
    index = condition.index
    for value in (when_true, when_false):
        if isinstance(value, pd.Series) and not value.index.equals(index):
            raise ValueError("WHERE 所有序列的索引必须完全一致")
    true_values = when_true if isinstance(when_true, pd.Series) else pd.Series(when_true, index=index)
    false_values = when_false if isinstance(when_false, pd.Series) else pd.Series(when_false, index=index)
    result = false_values.where(~condition.fillna(False).astype(bool), true_values)
    return result.where(condition.notna())


def ts_pct_change_safe(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return safe_div(series, series.shift(window)) - 1.0


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).mean()


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).sum()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).std(ddof=1)


def ts_median(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).median()


def ts_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    window = _window(window)
    if not 0.0 <= q <= 1.0:
        raise ValueError("q 必须在[0, 1]之间")
    return series.rolling(window, min_periods=window).quantile(q)


def ts_min(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).min()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).max()


def ts_skew(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).skew()


def ts_kurt(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).kurt()


def ts_position(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    low, high = ts_min(series, window), ts_max(series, window)
    result = safe_div(series - low, high - low)
    return result.mask(low.notna() & high.notna() & (high == low), 0.5)


def ts_rank_pct(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).apply(
        lambda values: pd.Series(values).rank(method="average", pct=True).iloc[-1], raw=True
    )


def ts_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    if not left.index.equals(right.index):
        raise ValueError("TS_CORR 两个输入的索引必须完全一致")
    return left.rolling(window, min_periods=window).corr(right)


def ts_cov(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    if not left.index.equals(right.index):
        raise ValueError("TS_COV 两个输入的索引必须完全一致")
    return left.rolling(window, min_periods=window).cov(right, ddof=1)


def ts_mad(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return series.rolling(window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))), raw=True
    )


def ewm_mean(series: pd.Series, span: int) -> pd.Series:
    span = _window(span)
    segment = series.isna().cumsum()
    result = series.groupby(segment, sort=False, group_keys=False).transform(
        lambda values: values.ewm(span=span, adjust=False, min_periods=span).mean()
    )
    return result.where(series.notna())


def _condition_values(condition: pd.Series) -> pd.Series:
    values = pd.to_numeric(condition, errors="coerce").astype(float)
    valid = values.dropna()
    if not valid.isin([0.0, 1.0]).all():
        raise ValueError("条件序列只能包含True/False、1/0或NaN")
    return values


def ts_count(condition: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    return _condition_values(condition).rolling(window, min_periods=window).sum()


def ts_consecutive_count(condition: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    values = _condition_values(condition)

    def trailing_count(items: np.ndarray) -> float:
        count = 0
        for item in items[::-1]:
            if item != 1.0:
                break
            count += 1
        return float(count)

    return values.rolling(window, min_periods=window).apply(trailing_count, raw=True)


def ts_rsquare(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    return ts_corr(left, right, window).pow(2)


def ts_slope(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    x = np.arange(window, dtype=float)
    return series.rolling(window, min_periods=window).apply(
        lambda values: np.polyfit(x, values, 1)[0], raw=True
    )


def ts_decay_linear(series: pd.Series, window: int) -> pd.Series:
    window = _window(window)
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights)), raw=True
    )


def clip_values(series: pd.Series, lower: float, upper: float) -> pd.Series:
    if lower > upper:
        raise ValueError("lower 不能大于 upper")
    return series.clip(lower=lower, upper=upper)


def cs_rank(series: pd.Series, dates: pd.Series) -> pd.Series:
    if len(series) != len(dates):
        raise ValueError("CS_RANK 输入长度不一致")
    if not series.index.equals(dates.index):
        raise ValueError("CS_RANK 输入索引必须完全一致")
    return series.groupby(dates, sort=False).rank(method="average", pct=True)


def cs_zscore(series: pd.Series, dates: pd.Series) -> pd.Series:
    if len(series) != len(dates):
        raise ValueError("CS_ZSCORE 输入长度不一致")
    if not series.index.equals(dates.index):
        raise ValueError("CS_ZSCORE 输入索引必须完全一致")
    grouped = series.groupby(dates, sort=False)
    return safe_div(series - grouped.transform("mean"), grouped.transform("std"))


def apply_by_symbol(
    series: pd.Series, symbols: pd.Series, operator: Callable[..., pd.Series],
    *args, dates: pd.Series, **kwargs,
) -> pd.Series:
    if len(series) != len(symbols) or len(series) != len(dates):
        raise ValueError("series、symbols与dates长度不一致")
    if not series.index.equals(symbols.index) or not series.index.equals(dates.index):
        raise ValueError("series、symbols与dates索引必须完全一致")
    frame = pd.DataFrame({"value": series, "symbol": symbols, "date": pd.to_datetime(dates)}, index=series.index)
    monotonic = frame.groupby("symbol", sort=False)["date"].apply(lambda values: values.is_monotonic_increasing)
    if not bool(monotonic.all()):
        raise ValueError("每只股票的数据必须按date升序排列")
    return frame.groupby("symbol", sort=False, group_keys=False)["value"].transform(
        lambda values: operator(values, *args, **kwargs)
    )


SAFE_OPERATORS: dict[str, Callable] = {
    "DELAY": ts_delay, "DELTA": ts_delta, "TS_PCT_CHANGE": ts_pct_change_safe,
    "TS_MEAN": ts_mean, "TS_SUM": ts_sum, "TS_STD": ts_std,
    "TS_MEDIAN": ts_median, "TS_QUANTILE": ts_quantile, "TS_MIN": ts_min,
    "TS_MAX": ts_max, "TS_SKEW": ts_skew, "TS_KURT": ts_kurt,
    "TS_POSITION": ts_position, "TS_RANK_PCT": ts_rank_pct, "TS_CORR": ts_corr,
    "TS_COV": ts_cov, "TS_MAD": ts_mad, "EWM_MEAN": ewm_mean,
    "TS_COUNT": ts_count, "TS_CONSECUTIVE_COUNT": ts_consecutive_count,
    "TS_RSQUARE": ts_rsquare, "TS_SLOPE": ts_slope,
    "TS_DECAY_LINEAR": ts_decay_linear, "SAFE_DIV": safe_div, "CLIP": clip_values,
    "ABS": abs_values, "SIGN": sign_values, "LOG1P": log1p_safe,
    "SQRT": sqrt_safe, "WHERE": where_values,
    "FIN_LAG_REPORT": fin_lag_report, "FIN_DELTA_REPORT": fin_delta_report,
    "CS_RANK": cs_rank, "CS_ZSCORE": cs_zscore,
}
