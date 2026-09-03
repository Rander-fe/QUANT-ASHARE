# -*- coding: utf-8 -*-
"""
量价交互因子（Volume-Price Interaction）

实现因子（共12个）：
    - RET_VOL_CORR_5/10/20: 收益率与成交量变化率的相关系数（量价背离的线性版本）
    - VWAP_DEV_5/10: 收盘价对VWAP的偏离 = (close - vwap_N) / vwap_N
    - AMOUNT_TREND_5/20: 成交额趋势 = amount / amount_MA_N
    - VOL_CV_20: 成交量变异系数 = std(volume) / mean(volume)
    - RET_AMT_CORR_20: 收益率与成交额相关系数
    - VOL_POS_10: 成交量在近10日的位置 = (vol - min) / (max - min)
    - MONEY_FLOW_20: 资金流强度 = sum(sign(ret) * amount, 20) / sum(amount, 20)

数据依赖：
    - close, volume, amount, vwap: 日线行情
    - 所有计算只用过去数据，无未来函数

设计原则：
    - 去量纲：相关系数/位置/比值均无量纲
    - 防除零：分母统一 +1e-12
"""

import numpy as np
import pandas as pd

from factors.registry import register
from factors.operators import ts_corr


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
# 1. 量价相关系数（RET_VOL_CORR）
# ============================================================
def calc_ret_vol_corr(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    收益率与成交量变化率的相关系数。
    值 > 0：价涨量增（配合良好）；值 < 0：量价背离。
    A股实证：量价背离（负相关）是A股最强单因子之一（ICIR > 0.6），
    因此本因子取负：负相关（背离）时因子值高，代表看涨反转信号。
    """
    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] volume_price: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    ret = _group_transform(df, "close", lambda s: s.pct_change())
    vol_chg = _group_transform(df, "volume", lambda s: s.pct_change())

    # 直接对构造好的序列按 symbol 分组，避免 pandas 2.x name=None 问题
    def _corr_group(x: pd.Series) -> pd.Series:
        return x.rolling(window, min_periods=window).corr(vol_chg.loc[x.index])

    corr = ret.groupby(df["symbol"]).transform(_corr_group)
    # 取负：背离（负相关）→ 高因子值（反转看涨）
    return -corr


def _register_ret_vol_corr_factors():
    """注册量价相关系数因子（5/10/20日）"""
    for w in [5, 10, 20]:
        register(
            name=f"RET_VOL_CORR{w}",
            category="liquidity",
            func=lambda df, w=w: calc_ret_vol_corr(df, w),
            comment=f"{w}日收益率与量变化率相关系数取负，背离越大因子值越高（A股最强因子族）"
        )


# ============================================================
# 2. VWAP偏离（VWAP_DEV）
# ============================================================
def calc_vwap_dev(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    收盘价对N日VWAP的偏离 = (close - vwap_N) / vwap_N
    VWAP_N = sum(amount, N) / sum(volume, N)（N日成交均价）
    值 > 0 表示收盘价高于N日平均成交价（买方占优）；< 0 表示低于（卖方占优）。
    A股特性：价格明显高于VWAP后易均值回归，负偏离是左侧买入信号。
    """
    required = ["close", "amount", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] volume_price: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc(group: pd.DataFrame) -> pd.Series:
        amt_n = group["amount"].rolling(window, min_periods=1).sum()
        vol_n = group["volume"].rolling(window, min_periods=1).sum()
        vwap_n = amt_n / (vol_n + 1e-12)
        return (group["close"] - vwap_n) / (vwap_n + 1e-12)

    return _group_apply(df, _calc)


def _register_vwap_dev_factors():
    """注册VWAP偏离因子（5/10日）"""
    for w in [5, 10]:
        register(
            name=f"VWAP_DEV{w}",
            category="technical",
            func=lambda df, w=w: calc_vwap_dev(df, w),
            comment=f"{w}日收盘价对VWAP偏离 = (close-vwap{w})/vwap{w}，负偏离左侧买入信号"
        )


# ============================================================
# 3. 成交额趋势（AMOUNT_TREND）
# ============================================================
def calc_amount_trend(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    成交额趋势 = amount / amount_MA_N
    > 1 表示近期成交额放大（资金活跃），< 1 表示萎缩（资金冷淡）。
    与成交量趋势类似，但用成交额可避免股本变化带来的量级偏差。
    """
    if "amount" not in df.columns:
        print("[WARN] volume_price: 缺少 amount 字段")
        return pd.Series(index=df.index, dtype=float)

    amount = df["amount"].fillna(0).astype(float)
    ma = _group_transform(df, "amount", lambda s: s.rolling(window, min_periods=1).mean())
    return amount / (ma + 1e-12)


def _register_amount_trend_factors():
    """注册成交额趋势因子（5/20日）"""
    for w in [5, 20]:
        register(
            name=f"AMOUNT_TREND{w}",
            category="liquidity",
            func=lambda df, w=w: calc_amount_trend(df, w),
            comment=f"{w}日成交额趋势 = amount/amount_MA{w}，>1放量资金活跃"
        )


# ============================================================
# 4. 成交量变异系数（VOL_CV）
# ============================================================
def calc_vol_cv(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    成交量变异系数 = std(volume, N) / mean(volume, N)
    衡量成交量波动的相对离散程度。
    值低 = 量能稳定（机构吸筹特征）；值高 = 量能忽大忽小（游资/情绪特征）。
    A股特性：量能稳定的股票更可能被机构控盘，取负后值越高越看好。
    """
    if "volume" not in df.columns:
        print("[WARN] volume_price: 缺少 volume 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc(series: pd.Series) -> pd.Series:
        mean_v = series.rolling(window, min_periods=1).mean()
        std_v = series.rolling(window, min_periods=1).std()
        return std_v / (mean_v + 1e-12)

    cv = df.groupby("symbol")["volume"].transform(_calc)
    # 取负：量能越稳定（CV低）因子值越高（机构控盘特征）
    return -cv


def _register_vol_cv_factors():
    """注册成交量变异系数因子（10/20日）"""
    for w in [10, 20]:
        register(
            name=f"VOL_CV{w}",
            category="liquidity",
            func=lambda df, w=w: calc_vol_cv(df, w),
            comment=f"{w}日成交量变异系数取负，量能稳定=机构控盘，值越高越看好"
        )


# ============================================================
# 5. 收益率-成交额相关（RET_AMT_CORR）
# ============================================================
def calc_ret_amt_corr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    收益率与成交额变化的相关系数（取负）。
    与 RET_VOL_CORR 互补：成交额维度捕捉资金进出，量维度捕捉交易活跃度。
    """
    required = ["close", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] volume_price: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    ret = _group_transform(df, "close", lambda s: s.pct_change())
    amt_chg = _group_transform(df, "amount", lambda s: s.pct_change())

    def _corr_group(x: pd.Series) -> pd.Series:
        return x.rolling(window, min_periods=window).corr(amt_chg.loc[x.index])

    corr = ret.groupby(df["symbol"]).transform(_corr_group)
    return -corr


def _register_ret_amt_corr_factors():
    """注册收益率-成交额相关因子（10/20日）"""
    for w in [10, 20]:
        register(
            name=f"RET_AMT_CORR{w}",
            category="liquidity",
            func=lambda df, w=w: calc_ret_amt_corr(df, w),
            comment=f"{w}日收益率与成交额变化相关取负，资金流向背离信号"
        )


# ============================================================
# 6. 成交量位置（VOL_POS）
# ============================================================
def calc_vol_pos(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    成交量在近N日的位置 = (vol - min) / (max - min)
    0 = 近N日最低量，1 = 近N日最高量。
    值高 = 近期放量（关注度上升）；值低 = 缩量（关注度下降）。
    """
    if "volume" not in df.columns:
        print("[WARN] volume_price: 缺少 volume 字段")
        return pd.Series(index=df.index, dtype=float)

    vol = df["volume"].fillna(0).astype(float)
    vol_max = _group_transform(df, "volume", lambda s: s.rolling(window, min_periods=1).max())
    vol_min = _group_transform(df, "volume", lambda s: s.rolling(window, min_periods=1).min())
    return (vol - vol_min) / (vol_max - vol_min + 1e-12)


def _register_vol_pos_factors():
    """注册成交量位置因子（10/20日）"""
    for w in [10, 20]:
        register(
            name=f"VOL_POS{w}",
            category="liquidity",
            func=lambda df, w=w: calc_vol_pos(df, w),
            comment=f"{w}日成交量在高低区间位置，衡量放量/缩量程度"
        )


# ============================================================
# 7. 资金流强度（MONEY_FLOW）
# ============================================================
def calc_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    资金流强度 = sum(sign(ret) * amount, N) / sum(amount, N)
    值 > 0：上涨日成交额占比高（资金流入主导）；< 0：下跌日占比高（流出主导）。
    取负后值越高代表资金持续流出（超卖反转信号）。
    """
    required = ["close", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] volume_price: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc(group: pd.DataFrame) -> pd.Series:
        ret = group["close"].pct_change()
        sign = np.sign(ret)
        flow = (sign * group["amount"]).rolling(window, min_periods=1).sum()
        total = group["amount"].rolling(window, min_periods=1).sum()
        return flow / (total + 1e-12)

    return _group_apply(df, _calc)


def _register_money_flow_factors():
    """注册资金流强度因子（10/20日）"""
    for w in [10, 20]:
        register(
            name=f"MONEY_FLOW{w}",
            category="liquidity",
            func=lambda df, w=w: calc_money_flow(df, w),
            comment=f"{w}日资金流强度 = sum(sign(ret)*amount)/sum(amount)，衡量资金进出方向"
        )


# 执行注册
_register_ret_vol_corr_factors()
_register_vwap_dev_factors()
_register_amount_trend_factors()
_register_vol_cv_factors()
_register_ret_amt_corr_factors()
_register_vol_pos_factors()
_register_money_flow_factors()
