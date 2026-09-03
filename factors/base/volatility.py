# -*- coding: utf-8 -*-
"""
波动率因子（Volatility）

实现因子（共10个）：
    - DOWN_STD_5/10/20/60: 下行波动率（负收益率的滚动标准差）
    - ATR_14: 14日平均真实波幅
    - HIGH_LOW_RANGE_20/60: 价格高低区间（通道宽度）
    - SKEW_20: 20日收益率偏度（取负：负偏度越大，因子值越高）
    - MAX_DRAWDOWN_20/60: 最大回撤（已在 reversal_momentum 中实现，此处不重复）

数据依赖：
    - high, low, close: 日线行情

A股波动率因子特性：
    - 低波动率在A股中长期具有显著正溢价（低波异象）
    - 下行波动率比整体波动率更具预测能力
    - 偏度因子（负偏）在A股中具有较强反转信号
"""

import numpy as np
import pandas as pd

from factors.registry import register


def calc_downside_std(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    下行波动率 = 负收益率的滚动标准差
    只计算负收益率（ret < 0）的标准差，衡量下行风险
    值越高，代表下行风险越大（因子取负，值越高代表下行风险越低）
    """
    if "close" not in df.columns:
        print("[WARN] volatility: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ret = series.pct_change()
        # 只取负收益率，正收益率置为 0（不参与下行波动计算）
        negative_ret = ret.where(ret < 0, 0)
        # 滚动标准差（只用过去数据）
        downside_std = negative_ret.rolling(period, min_periods=1).std()
        return downside_std

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_downside_std_neg(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    下行波动率取负 = -downside_std
    值越高代表下行风险越低（低波动溢价）
    """
    series = calc_downside_std(df, period)
    return -series


def calc_atr_14(df: pd.DataFrame) -> pd.Series:
    """
    平均真实波幅（ATR），14日滚动平均
    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = 14日TR的移动平均

    业务含义：ATR 越高，代表近期价格波动越大
    因子取负：低ATR股票具有低波溢价
    """
    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] volatility: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        prev_close = group["close"].shift(1)

        # 计算真实波幅的三个分量
        tr1 = group["high"] - group["low"]
        tr2 = (group["high"] - prev_close).abs()
        tr3 = (group["low"] - prev_close).abs()

        # 取最大值
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 14日滚动平均（只用过去数据）
        atr = tr.rolling(14, min_periods=1).mean()
        return atr

    # include_groups=False: 避免 pandas 2.x 对分组列操作的 FutureWarning
    return df.groupby("symbol", group_keys=False).apply(_calc_group, include_groups=False).reset_index(level=0, drop=True)


def calc_atr_neg_14(df: pd.DataFrame) -> pd.Series:
    """
    平均真实波幅取负 = -ATR_14
    值越高代表波动越低（低波溢价）
    """
    series = calc_atr_14(df)
    return -series


def calc_high_low_range(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    价格高低区间 = (high_N - low_N) / (low_N + 1e-12)
    衡量过去N日内价格波动的绝对幅度（相对于最低价）

    值越高代表近期波动越剧烈
    """
    required = ["high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] volatility: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        high_n = group["high"].rolling(period, min_periods=1).max()
        low_n = group["low"].rolling(period, min_periods=1).min()
        range_val = (high_n - low_n) / (low_n + 1e-12)
        return range_val

    # include_groups=False: 避免 pandas 2.x 对分组列操作的 FutureWarning
    return df.groupby("symbol", group_keys=False).apply(_calc_group, include_groups=False).reset_index(level=0, drop=True)


def calc_high_low_range_neg(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    价格高低区间取负 = -range
    值越高代表波动越低（低波溢价）
    """
    series = calc_high_low_range(df, period)
    return -series


def calc_skew_20(df: pd.DataFrame) -> pd.Series:
    """
    20日收益率偏度
    偏度 < 0 表示左偏（尾部下行风险大），偏度 > 0 表示右偏

    因子取负：负偏度越大（左偏越严重），因子值越高，
    意味着该股票近期经历过极端下跌，未来可能出现反转
    """
    if "close" not in df.columns:
        print("[WARN] volatility: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ret = series.pct_change()
        skew_20 = ret.rolling(20, min_periods=5).skew()
        return skew_20

    skew = df.groupby("symbol")["close"].transform(_calc_group)
    # 取负：负偏度（左偏）转换为高因子值
    return -skew


def _register_volatility_factors():
    """注册波动率因子（共10个）"""
    # 下行波动率（5/10/20/60日）
    for w in [5, 10, 20, 60]:
        register(
            name=f"DOWN_STD_{w}",
            category="volatility",
            func=lambda df, w=w: calc_downside_std(df, w),
            comment=f"{w}日下行波动率（负收益率标准差），值越高下行风险越大"
        )
        register(
            name=f"LOW_DOWN_STD_{w}",
            category="volatility",
            func=lambda df, w=w: -calc_downside_std(df, w),
            comment=f"{w}日低下行波动率，值越高代表下行风险越低（低波溢价）"
        )

    # ATR取负
    register(
        name="LOW_ATR_14",
        category="volatility",
        func=calc_atr_neg_14,
        comment="14日平均真实波幅取负，值越高代表波动越低（低波溢价）"
    )

    # 高低区间取负（20/60日）
    for w in [20, 60]:
        register(
            name=f"LOW_RANGE_{w}",
            category="volatility",
            func=lambda df, w=w: -calc_high_low_range(df, w),
            comment=f"{w}日价格区间取负，值越高代表波动越低（低波溢价）"
        )

    # 负偏度因子
    register(
        name="NEG_SKEW_20",
        category="volatility",
        func=calc_skew_20,
        comment="20日收益率偏度取负，负偏度越大（左尾风险）因子值越高，反转信号"
    )

    # 整体波动率取负（5/10/20/60日）——低波异象，与下行波动率互补
    for w in [5, 10, 20, 60]:
        register(
            name=f"STD_RET_{w}",
            category="volatility",
            func=lambda df, w=w: _neg_std_ret(df, w),
            comment=f"{w}日整体收益率标准差取负，低波溢价（与LOW_DOWN_STD互补）"
        )

    # 波动率的波动（VOL_OF_VOL_20）
    register(
        name="VOL_OF_VOL_20",
        category="volatility",
        func=calc_vol_of_vol_20,
        comment="20日波动率的波动取负，波动率稳定性高者因子值高（机构控盘特征）"
    )

    # ATR 相对价格比例取负（ATR_PCT_14）
    register(
        name="LOW_ATR_PCT_14",
        category="volatility",
        func=calc_atr_pct_neg_14,
        comment="14日ATR占价格比例取负，消除价格水平差异后的低波因子"
    )


def _neg_std_ret(df: pd.DataFrame, period: int) -> pd.Series:
    """整体收益率标准差取负（低波溢价）：-std(daily_ret, N)"""
    if "close" not in df.columns:
        print("[WARN] volatility: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ret = series.pct_change()
        return -ret.rolling(period, min_periods=period).std()

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_vol_of_vol_20(df: pd.DataFrame) -> pd.Series:
    """
    波动率的波动（VOL_OF_VOL）= -std(rolling_std_20, 20)。

    衡量波动率的稳定性：值越高代表波动率越平稳（量价关系稳定，
    更可能是机构有序控盘而非游资情绪驱动）。
    """
    if "close" not in df.columns:
        print("[WARN] volatility: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ret = series.pct_change()
        vol20 = ret.rolling(20, min_periods=5).std()
        return -vol20.rolling(20, min_periods=5).std()

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_atr_pct_neg_14(df: pd.DataFrame) -> pd.Series:
    """
    ATR 相对价格比例取负 = -ATR_14 / close。

    与 LOW_ATR_14（绝对ATR取负）不同：除以价格后消除价格水平差异，
    高价股与低价股的波动率可直接比较。
    """
    atr = calc_atr_14(df)
    close = df["close"].astype(float)
    return -atr / (close + 1e-12)


# 执行注册（模块导入时自动调用）
_register_volatility_factors()