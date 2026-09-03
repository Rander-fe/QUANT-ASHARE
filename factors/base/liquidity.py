# -*- coding: utf-8 -*-
"""
流动性因子（Liquidity）

实现因子：
    - LOW_TURNOVER: 低换手因子（换手率取负，越高代表换手越低）
    - LN_TURNOVER: 换手率对数（log(turnover_rate)，处理极端值）
    - AMIHUD_20: Amihud 非流动性指标（20日平均，衡量价格冲击成本）
    - VOL_TREND: 量能趋势（当前成交量 / 过去20日均量）
    - VOL_PRICE_DIVERGENCE: 量价背离（价格与成交量的方向一致性）

数据依赖：
    - turnover_rate: 来自 daily_basic
    - close / volume / amount: 来自日线行情

注意：
    - 所有滚动窗口计算使用 rolling(min_periods=1)，避免早期数据全为 NaN
    - 因子值不进行缩尾/标准化，由后续 Qlib DataHandler 处理
"""

import numpy as np
import pandas as pd

from factors.registry import register


def calc_low_turnover(df: pd.DataFrame) -> pd.Series:
    """
    低换手因子：换手率取负值（-turnover_rate）
    A股实证：低换手率组合具有显著正超额收益
    """
    if "turnover_rate" not in df.columns:
        print("[WARN] liquidity: 缺少 turnover_rate 字段")
        return pd.Series(index=df.index, dtype=float)
    # 处理缺失值和极端值（换手率不会为负，但可能有 0 或 NaN）
    series = df["turnover_rate"].fillna(0).astype(float)
    # 取负：换手率越低，因子值越高
    return -series


def calc_ln_turnover(df: pd.DataFrame) -> pd.Series:
    """
    换手率对数：log(turnover_rate + 1e-6)
    处理换手率分布的右偏，使其更接近正态分布
    """
    if "turnover_rate" not in df.columns:
        print("[WARN] liquidity: 缺少 turnover_rate 字段")
        return pd.Series(index=df.index, dtype=float)
    series = df["turnover_rate"].fillna(0).astype(float)
    # 加小常数防止 log(0)
    return np.log(series + 1e-6)


def calc_amihud_20(df: pd.DataFrame) -> pd.Series:
    """
    Amihud 非流动性指标（20日滚动平均）
    定义：Amihud = mean(|日收益率| / 日成交额)
    值越大，表示流动性越差（价格冲击成本越高）
    A股实证：高 Amihud 股票通常具有溢价（但流动性差的股票也有较高风险）
    """
    required = ["close", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] liquidity: 缺少字段 {missing}，无法计算 Amihud")
        return pd.Series(index=df.index, dtype=float)

    # 计算日收益率
    ret = df.groupby("symbol")["close"].transform(lambda x: x.pct_change())

    # 计算每日 Amihud 值：|ret| / (amount + 1e-12)
    # 注意：pandas 2.x 下 Series 运算后 name 为 None，不能依赖 name 从 df 取列，
    # 直接对 amihud_daily 本身按 symbol 分组滚动（索引与 df 对齐）。
    amihud_daily = ret.abs() / (df["amount"].fillna(0).astype(float) + 1e-12)

    # 20日滚动平均（只用过去数据）
    amihud_20 = amihud_daily.groupby(df["symbol"]).transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    return amihud_20


def calc_vol_trend(df: pd.DataFrame) -> pd.Series:
    """
    量能趋势：当前成交量 / 过去20日均量
    > 1 表示放量，< 1 表示缩量
    结合价格变化可衍生出量价配合信号
    """
    if "volume" not in df.columns:
        print("[WARN] liquidity: 缺少 volume 字段")
        return pd.Series(index=df.index, dtype=float)

    volume = df["volume"].fillna(0).astype(float)

    # 20日均量（只用过去数据）
    vol_ma_20 = df.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )

    return volume / (vol_ma_20 + 1e-12)


def calc_vol_price_divergence(df: pd.DataFrame) -> pd.Series:
    """
    量价背离因子：价格变化与成交量变化的相关系数（负相关时因子值高）
    逻辑：价涨量缩（背离）或价跌量放（背离）是重要的反转信号

    计算方法：计算过去5日价格变化与成交量变化的 Spearman 秩相关系数
    如果相关系数为负（量价背离），因子值高
    """
    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] liquidity: 缺少字段 {missing}，无法计算量价背离")
        return pd.Series(index=df.index, dtype=float)

    # 计算日收益率和成交量变化率
    ret = df.groupby("symbol")["close"].transform(lambda x: x.pct_change())
    vol_chg = df.groupby("symbol")["volume"].transform(lambda x: x.pct_change())

    # 使用 rolling window 计算每5日的 Spearman 相关系数
    # 注意：这里使用 apply 实现，但数据量较大时可能较慢
    def _rolling_corr_5(x, y):
        """计算过去5个值的秩相关系数（两个序列长度必须 ≥ 5）"""
        if len(x) < 5:
            return np.nan
        return x.corr(y, method="spearman")

    # 为减少计算量，我们使用简单的方向一致性指标代替完整秩相关
    # 替代方案：计算过去5日价格变动方向与成交量变动方向的一致性比率
    # 这里采用更轻量的实现：直接计算每日价格变化与成交量变化的符号一致性
    # 1 表示方向一致，-1 表示背离，然后取过去5日平均值
    sign_ret = np.sign(ret)
    sign_vol = np.sign(vol_chg)

    # 一致性 = 1 如果符号相同，否则 -1
    consistency = (sign_ret == sign_vol).astype(int) * 2 - 1

    # 5日滚动平均一致性（只用过去数据）
    # 同 calc_amihud_20：直接对 consistency 按 symbol 分组，避免依赖丢失的 name
    div_5 = consistency.groupby(df["symbol"]).transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # 量价背离 = 负的一致性（一致性越低，背离越严重，因子值越高）
    return -div_5


def _register_liquidity_factors():
    """注册流动性因子"""
    register(
        name="LOW_TURNOVER",
        category="liquidity",
        func=calc_low_turnover,
        comment="低换手因子：-turnover_rate，A股第二强Alpha来源"
    )
    register(
        name="LN_TURNOVER",
        category="liquidity",
        func=calc_ln_turnover,
        comment="换手率对数：log(turnover_rate + 1e-6)，处理右偏分布"
    )
    register(
        name="AMIHUD_20",
        category="liquidity",
        func=calc_amihud_20,
        comment="Amihud非流动性指标：20日平均 |ret|/amount，衡量价格冲击成本"
    )
    register(
        name="VOL_TREND",
        category="liquidity",
        func=calc_vol_trend,
        comment="量能趋势：volume / volume_ma_20，>1放量，<1缩量"
    )
    register(
        name="VOL_PRICE_DIVERGENCE",
        category="liquidity",
        func=calc_vol_price_divergence,
        comment="量价背离因子：价格与成交量方向一致性取负，背离越大因子值越高"
    )


# 执行注册（模块导入时自动调用）
_register_liquidity_factors()