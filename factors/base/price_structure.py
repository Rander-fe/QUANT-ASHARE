# -*- coding: utf-8 -*-
"""
价格结构因子（Price Structure）

实现因子（共18个）：
    - BIAS_5/10/20/60: 乖离率 = (close - MA_N) / MA_N，价格偏离均线的程度
    - POS_MA_5/10/20: 价格在N日均线上的相对位置 = (close - min_N) / (max_N - min_N)
    - UPPER_SHADOW_10/20: 上影线比例 = (high - max(open,close)) / (high - low)
    - LOWER_SHADOW_10/20: 下影线比例 = (min(open,close) - low) / (high - low)
    - GAP_5/10: 跳空缺口 = (open - prev_close) / prev_close
    - MA_ALIGN_5_10_20: 均线多头排列强度（短期在长期之上为正）
    - CLOSE_POS_5/20: 收盘价在近N日最高最低区间的位置（0=最低，1=最高）

数据依赖：
    - open, high, low, close: 日线行情
    - 所有计算只用过去数据，无未来函数

设计原则：
    - 去量纲：所有因子除以价格或价格区间，消除价格水平差异
    - 防除零：分母统一 +1e-12
    - 命名：算子+窗口
"""

import pandas as pd

from factors.registry import register
from factors.operators import ts_ma, ts_max, ts_min


def _group_transform(df: pd.DataFrame, col: str, func) -> pd.Series:
    """按股票分组计算，返回与 df 行对齐的序列"""
    return df.groupby("symbol")[col].transform(func)


def _group_apply(df: pd.DataFrame, func) -> pd.Series:
    """按股票分组 DataFrame.apply（含 include_groups=False 防 FutureWarning）"""
    return (
        df.groupby("symbol", group_keys=False)
        .apply(func, include_groups=False)
        .reset_index(level=0, drop=True)
    )


# ============================================================
# 1. 乖离率（BIAS）
# ============================================================
def calc_bias(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    乖离率 = (close - MA_N) / MA_N
    衡量价格偏离其N日均线的程度。
    值 > 0 表示价格在均线上方（超买），值 < 0 表示价格在均线下方（超卖）。
    A股特性：乖离率过大后常出现均值回归（反转），负乖离是左侧买入信号。
    """
    if "close" not in df.columns:
        print("[WARN] price_structure: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    close = df["close"]
    ma = _group_transform(df, "close", lambda s: ts_ma(s, window))
    return (close - ma) / (ma.abs() + 1e-12)


def _register_bias_factors():
    """注册乖离率因子（5/10/20/60日）"""
    for w in [5, 10, 20, 60]:
        register(
            name=f"BIAS{w}",
            category="technical",
            func=lambda df, w=w: calc_bias(df, w),
            comment=f"{w}日乖离率 = (close - MA{w})/MA{w}，价格偏离均线程度，负乖离是反转买入信号"
        )


# ============================================================
# 2. 价格区间位置（POS_MA）
# ============================================================
def calc_pos_ma(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    价格在近N日高低区间中的位置 = (close - min_N) / (max_N - min_N)
    0 = 近N日最低价，1 = 近N日最高价。
    值越高代表价格越接近区间上沿（强势），越低代表越接近下沿（弱势）。
    """
    required = ["close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] price_structure: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    high_n = _group_transform(df, "high", lambda s: ts_max(s, window))
    low_n = _group_transform(df, "low", lambda s: ts_min(s, window))
    return (df["close"] - low_n) / (high_n - low_n + 1e-12)


def _register_pos_ma_factors():
    """注册价格位置因子（5/10/20日）"""
    for w in [5, 10, 20]:
        register(
            name=f"POS_MA{w}",
            category="technical",
            func=lambda df, w=w: calc_pos_ma(df, w),
            comment=f"{w}日价格在高低区间位置 = (close-min{w})/(max{w}-min{w})，0~1越接近1越强势"
        )


# ============================================================
# 3. 影线比例（上/下影线）
# ============================================================
def calc_upper_shadow(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    上影线比例 = (high - max(open, close)) / (high - low)
    衡量每根K线的上影线长度占比，再取N日均值。
    上影线长 = 上方抛压重 = 潜在反转下跌信号（取负后值越高越看好）。
    """
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] price_structure: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc(group: pd.DataFrame) -> pd.Series:
        body_top = group[["open", "close"]].max(axis=1)
        rng = group["high"] - group["low"]
        shadow = (group["high"] - body_top) / (rng + 1e-12)
        return shadow.rolling(window, min_periods=1).mean()

    return _group_apply(df, _calc)


def calc_lower_shadow(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    下影线比例 = (min(open, close) - low) / (high - low)
    衡量每根K线的下影线长度占比，再取N日均值。
    下影线长 = 下方承接力强 = 潜在反转上涨信号（值越高越看好）。
    """
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] price_structure: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc(group: pd.DataFrame) -> pd.Series:
        body_bottom = group[["open", "close"]].min(axis=1)
        rng = group["high"] - group["low"]
        shadow = (body_bottom - group["low"]) / (rng + 1e-12)
        return shadow.rolling(window, min_periods=1).mean()

    return _group_apply(df, _calc)


def _register_shadow_factors():
    """注册影线因子（上影10/20，下影10/20）"""
    for w in [10, 20]:
        register(
            name=f"UPPER_SHADOW{w}",
            category="technical",
            func=lambda df, w=w: calc_upper_shadow(df, w),
            comment=f"{w}日上影线比例均值，上影线长=抛压重，取负信号反转"
        )
        register(
            name=f"LOWER_SHADOW{w}",
            category="technical",
            func=lambda df, w=w: calc_lower_shadow(df, w),
            comment=f"{w}日下影线比例均值，下影线长=承接强，看涨信号"
        )


# ============================================================
# 4. 跳空缺口（GAP）
# ============================================================
def calc_gap(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    跳空缺口 = (open - prev_close) / prev_close
    衡量开盘相对昨收的跳空幅度，再取N日均值。
    向上跳空大 = 情绪高涨（可能追高风险）；向下跳空大 = 恐慌（可能超跌反转）。
    """
    required = ["open", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] price_structure: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc(group: pd.DataFrame) -> pd.Series:
        prev_close = group["close"].shift(1)
        gap = (group["open"] - prev_close) / (prev_close + 1e-12)
        return gap.rolling(window, min_periods=1).mean()

    return _group_apply(df, _calc)


def _register_gap_factors():
    """注册缺口因子（5/10日）"""
    for w in [5, 10]:
        register(
            name=f"GAP{w}",
            category="reversal_momentum",
            func=lambda df, w=w: calc_gap(df, w),
            comment=f"{w}日平均跳空缺口 = mean((open-prev_close)/prev_close)，衡量情绪/恐慌"
        )


# ============================================================
# 5. 均线排列（MA_ALIGN）
# ============================================================
def calc_ma_align(df: pd.DataFrame) -> pd.Series:
    """
    均线多头排列强度 = (MA5 - MA20) / (MA20 + 1e-12)
    值 > 0 表示短期均线在长期均线上方（多头排列，趋势向上）。
    值 < 0 表示空头排列（趋势向下）。
    A股特性：均线排列在趋势市中有效，震荡市中信号弱。
    """
    if "close" not in df.columns:
        print("[WARN] price_structure: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    ma5 = _group_transform(df, "close", lambda s: ts_ma(s, 5))
    ma20 = _group_transform(df, "close", lambda s: ts_ma(s, 20))
    return (ma5 - ma20) / (ma20 + 1e-12)


def _register_ma_align_factors():
    """注册均线排列因子"""
    register(
        name="MA_ALIGN_5_20",
        category="reversal_momentum",
        func=calc_ma_align,
        comment="均线排列强度 = (MA5-MA20)/MA20，>0多头排列，趋势信号"
    )


# ============================================================
# 6. 收盘价位置（CLOSE_POS）
# ============================================================
def calc_close_pos(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    收盘价在近N日高低区间的位置 = (close - min_N) / (max_N - min_N)
    与 POS_MA 类似但用收盘价，衡量价格在区间内的相对强弱。
    """
    required = ["close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] price_structure: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    high_n = _group_transform(df, "high", lambda s: ts_max(s, window))
    low_n = _group_transform(df, "low", lambda s: ts_min(s, window))
    return (df["close"] - low_n) / (high_n - low_n + 1e-12)


def _register_close_pos_factors():
    """注册收盘价位置因子（5/20日）"""
    for w in [5, 20]:
        register(
            name=f"CLOSE_POS{w}",
            category="technical",
            func=lambda df, w=w: calc_close_pos(df, w),
            comment=f"{w}日收盘价在高低区间位置，衡量收盘相对强弱"
        )


# 执行注册
_register_bias_factors()
_register_pos_ma_factors()
_register_shadow_factors()
_register_gap_factors()
_register_ma_align_factors()
_register_close_pos_factors()
