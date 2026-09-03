# -*- coding: utf-8 -*-
"""
横截面工具函数（需配合 groupby 使用）

提供因子计算中常用的横截面操作：
    - cs_rank: 横截面百分位排名
    - cs_zscore: 横截面标准化
    - winsorize: 横截面缩尾处理

⚠️ 重要：这些函数本身不分组，必须配合 groupby("date") 使用。
    错误用法：cs_zscore(df["factor"])  # ❌ 用全市场均值/方差
    正确用法：df.groupby("date")["factor"].apply(cs_zscore)  # ✅ 每日横截面
"""

import numpy as np
import pandas as pd


def cs_rank(series: pd.Series, pct: bool = True) -> pd.Series:
    """
    横截面百分位排名。

    Parameters:
        series: 待排名的序列（建议已按日期分组后的子集）
        pct: 是否返回 0~1 之间的百分位（默认 True）

    用法（需配合 groupby）：
        df["rank"] = df.groupby("date")[factor].apply(cs_rank)
    """
    if pct:
        return series.rank(pct=True)
    return series.rank()


def cs_zscore(series: pd.Series) -> pd.Series:
    """
    横截面 Z-Score 标准化。

    公式: (x - mean) / std

    用法（需配合 groupby）：
        df["zscore"] = df.groupby("date")[factor].apply(cs_zscore)
    """
    mean = series.mean()
    std = series.std()
    if std < 1e-12:
        return series * 0
    return (series - mean) / std


def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    横截面缩尾处理。

    Parameters:
        series: 待处理的序列
        limits: (lower, upper) 分位数，默认 (1%, 99%)

    用法（需配合 groupby）：
        df["winsorized"] = df.groupby("date")[factor].apply(winsorize)
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower=lower, upper=upper)