# factors/operators.py (修复版)
import numpy as np
import pandas as pd

# 强制 min_periods = window，避免早期截断干扰
def ts_pct_change(series, window):
    return series.pct_change(window)

def ts_ma(series, window):
    return series.rolling(window, min_periods=window).mean()

def ts_std(series, window):
    return series.rolling(window, min_periods=window).std()

def ts_max(series, window):
    return series.rolling(window, min_periods=window).max()

def ts_min(series, window):
    return series.rolling(window, min_periods=window).min()

def ts_rank(series, window):
    # 用 min_periods 保证窗口完整，rank 按位置计算
    return series.rolling(window, min_periods=window).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-12), raw=False
    )

def ts_corr(series_x, series_y, window):
    # 双序列必须保证索引对齐
    return series_x.rolling(window, min_periods=window).corr(series_y)

def ts_rsquare(series_x, series_y, window):
    def calc_r2(x, y):
        if len(x) < 2:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return corr ** 2
    return series_x.rolling(window, min_periods=window).apply(
        lambda x: calc_r2(x, series_y.loc[x.index]), raw=False
    )

def ts_slope(series, window):
    # 用向量化方式替代 Python 循环（高性能）
    def calc_slope(x):
        if len(x) < 2:
            return 0.0
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    return series.rolling(window, min_periods=window).apply(
        lambda x: calc_slope(x), raw=False
    )

# 注意：ts_decay_linear 保留但添加 min_periods 限制
def ts_decay_linear(series, window):
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    def _weighted_avg(x):
        if len(x) < window:
            return np.nan
        return np.dot(x, weights)
    return series.rolling(window, min_periods=window).apply(_weighted_avg, raw=True)

OPERATORS = {
    "ROC": ts_pct_change,
    "MA": ts_ma,
    "STD": ts_std,
    "MAX": ts_max,
    "MIN": ts_min,
    "RANK": ts_rank,
    "CORR": ts_corr,
    "RSQ": ts_rsquare,
    "SLOPE": ts_slope,
    "DECAY": ts_decay_linear,
}
# ============================================================
# Ref 算子（防未来函数）
# ============================================================

def ref(series: pd.Series, d: int) -> pd.Series:
    """获取过去第 d 日的值（d > 0）"""
    if not isinstance(d, int):
        raise TypeError(f"d must be int, got {type(d).__name__}")
    if d <= 0:
        raise ValueError(f"Ref 要求 d > 0，但 d={d}")
    return series.shift(d)