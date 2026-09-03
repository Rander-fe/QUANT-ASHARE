# -*- coding: utf-8 -*-
"""华泰《基于遗传规划的选股因子挖掘》六因子复现。

公式严格对应报告图表 12。所有时序算子均按 symbol 分组且只使用当前及
过去数据；``rank`` 是逐日横截面百分位排名。为避免除零和无穷值，除法
采用 1e-12 保护，最终将非有限值置为 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.registry import register

EPS = 1e-12
FACTOR_NAMES = tuple(f"GP_HT_ALPHA{i}" for i in range(1, 7))


def _finite(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)


def _group_rolling(
    values: pd.Series,
    symbols: pd.Series,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = values.groupby(symbols, sort=False)
    if operation == "sum":
        return grouped.transform(lambda s: s.rolling(window, min_periods=window).sum())
    if operation == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=window).std())
    raise ValueError(f"未知滚动操作: {operation}")


def _rolling_pair(
    left: pd.Series,
    right: pd.Series,
    symbols: pd.Series,
    window: int,
    operation: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for _, index in left.groupby(symbols, sort=False).groups.items():
        x = left.loc[index]
        y = right.loc[index]
        if operation == "corr":
            values = x.rolling(window, min_periods=window).corr(y)
        elif operation == "cov":
            values = x.rolling(window, min_periods=window).cov(y)
        else:
            raise ValueError(f"未知双变量滚动操作: {operation}")
        result.loc[index] = values.to_numpy()
    return result


def _cs_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return values.groupby(dates, sort=False).rank(method="average", pct=True)


def _daily_vwap(df: pd.DataFrame) -> pd.Series:
    """优先使用日频 VWAP；缺失时用 amount/volume 构造。

    数据源的 amount 可能带固定单位倍数。Alpha1只对 VWAP/HIGH 做时序
    相关，固定正比例不改变相关系数，因此不额外猜测单位。
    """
    if "vwap" in df.columns:
        vwap = pd.to_numeric(df["vwap"], errors="coerce")
    else:
        vwap = pd.to_numeric(df["amount"], errors="coerce") / (
            pd.to_numeric(df["volume"], errors="coerce") + EPS
        )
    return _finite(vwap)


def calc_gp_ht_alpha1(df: pd.DataFrame) -> pd.Series:
    """correlation(div(vwap, high), high, 10)。"""
    high = pd.to_numeric(df["high"], errors="coerce")
    ratio = _daily_vwap(df) / (high + EPS)
    return _finite(_rolling_pair(ratio, high, df["symbol"], 10, "corr"))


def calc_gp_ht_alpha2(df: pd.DataFrame) -> pd.Series:
    """ts_sum(rank(correlation(high, low, 20)), 20)。"""
    corr = _rolling_pair(df["high"], df["low"], df["symbol"], 20, "corr")
    ranked = _cs_rank(corr, df["date"])
    return _finite(_group_rolling(ranked, df["symbol"], 20, "sum"))


def calc_gp_ht_alpha3(df: pd.DataFrame) -> pd.Series:
    """-ts_stddev(volume, 5)。"""
    return _finite(-_group_rolling(df["volume"], df["symbol"], 5, "std"))


def calc_gp_ht_alpha4(df: pd.DataFrame) -> pd.Series:
    """-rank(covariance(high, volume, 10))*rank(ts_stddev(high, 10))。"""
    cov = _rolling_pair(df["high"], df["volume"], df["symbol"], 10, "cov")
    high_std = _group_rolling(df["high"], df["symbol"], 10, "std")
    return _finite(-_cs_rank(cov, df["date"]) * _cs_rank(high_std, df["date"]))


def calc_gp_ht_alpha5(df: pd.DataFrame) -> pd.Series:
    """-ts_sum(rank(covariance(high,volume,5)),5)*rank(ts_stddev(high,5))。"""
    cov = _rolling_pair(df["high"], df["volume"], df["symbol"], 5, "cov")
    cov_rank = _cs_rank(cov, df["date"])
    cov_rank_sum = _group_rolling(cov_rank, df["symbol"], 5, "sum")
    high_std = _group_rolling(df["high"], df["symbol"], 5, "std")
    return _finite(-cov_rank_sum * _cs_rank(high_std, df["date"]))


def calc_gp_ht_alpha6(df: pd.DataFrame) -> pd.Series:
    """ts_sum(div(add(high, low), close), 5)。

    报告图表 12 的排版括号错位；按函数定义还原为上述合法表达式。
    """
    ratio = (df["high"] + df["low"]) / (df["close"] + EPS)
    return _finite(_group_rolling(ratio, df["symbol"], 5, "sum"))


register("GP_HT_ALPHA1", "composite", calc_gp_ht_alpha1,
         "华泰GP复现1：VWAP/HIGH与HIGH的10日相关")
register("GP_HT_ALPHA2", "composite", calc_gp_ht_alpha2,
         "华泰GP复现2：HIGH/LOW 20日相关的截面排名再做20日求和")
register("GP_HT_ALPHA3", "composite", calc_gp_ht_alpha3,
         "华泰GP复现3：5日成交量标准差取负")
register("GP_HT_ALPHA4", "composite", calc_gp_ht_alpha4,
         "华泰GP复现4：10日高价-成交量协方差排名与高价波动排名复合")
register("GP_HT_ALPHA5", "composite", calc_gp_ht_alpha5,
         "华泰GP复现5：5日量价协方差排名和与高价波动排名复合")
register("GP_HT_ALPHA6", "composite", calc_gp_ht_alpha6,
         "华泰GP复现6：(HIGH+LOW)/CLOSE的5日求和")

