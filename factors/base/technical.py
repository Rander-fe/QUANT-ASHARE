# -*- coding: utf-8 -*-
"""
技术指标因子（Technical Indicators）

实现因子（共11个）：
    - RSI_14: 14日相对强弱指标
    - RSI_21: 21日相对强弱指标（更长周期）
    - MACD_LINE: 12日EMA - 26日EMA（MACD线）
    - MACD_HIST: MACD线 - 信号线（MACD柱状图）
    - KDJ_J: 9日KDJ的J值
    - CCI_20: 20日商品通道指标
    - BB_BANDWIDTH: 布林带带宽（20日，2倍标准差）
    - BB_POSITION: 价格在布林带中的相对位置
    - OBV_CHG_20: 能量潮20日变化率

数据依赖：
    - open, high, low, close, volume: 日线行情
    - 所有计算只用过去数据，无未来函数
"""

import numpy as np
import pandas as pd

from factors.registry import register


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    相对强弱指标（RSI）
    RSI = 100 - (100 / (1 + RS))
    RS = 平均涨幅 / 平均跌幅

    业务含义：RSI > 70 为超买，RSI < 30 为超卖
    A股特性：RSI 在震荡市中效果较好，单边市中容易钝化
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_macd_line(df: pd.DataFrame) -> pd.Series:
    """
    MACD线 = 12日EMA - 26日EMA
    当MACD线 > 0 时，短期均线在长期均线之上（多头趋势）
    当MACD线从下向上穿过0轴为买入信号
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ema_12 = series.ewm(span=12, min_periods=1, adjust=False).mean()
        ema_26 = series.ewm(span=26, min_periods=1, adjust=False).mean()
        return ema_12 - ema_26

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_macd_hist(df: pd.DataFrame) -> pd.Series:
    """
    MACD柱状图 = MACD线 - 信号线（9日EMA of MACD线）
    柱状图 > 0 表示多头动能增强，< 0 表示空头动能增强
    柱状图由负转正为买入信号，由正转负为卖出信号
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ema_12 = series.ewm(span=12, min_periods=1, adjust=False).mean()
        ema_26 = series.ewm(span=26, min_periods=1, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, min_periods=1, adjust=False).mean()
        return macd_line - signal_line

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_kdj_j(df: pd.DataFrame, period: int = 9) -> pd.Series:
    """
    KDJ随机指标 - J值
    RSV = (close - low_N) / (high_N - low_N) * 100
    K = 2/3 * K_prev + 1/3 * RSV  (K初始值50)
    D = 2/3 * D_prev + 1/3 * K
    J = 3K - 2D

    J值 > 100 为超买，J值 < 0 为超卖
    使用简单的滚动窗口近似计算（非递归平滑）
    """
    required = ["close", "high", "low"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] technical: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        # 计算RSV：使用滚动窗口的最近N日最高/最低
        high_roll = group["high"].rolling(period, min_periods=1)
        low_roll = group["low"].rolling(period, min_periods=1)
        high_n = high_roll.max()
        low_n = low_roll.min()

        rsv = (group["close"] - low_n) / (high_n - low_n + 1e-12) * 100
        rsv = rsv.fillna(50)  # 早期数据用50填充

        # 近似计算K和D（使用指数加权平均模拟平滑效果）
        # 为了简化，使用rolling mean来近似
        k = rsv.rolling(3, min_periods=1).mean()
        d = k.rolling(3, min_periods=1).mean()
        j = 3 * k - 2 * d
        return j

    # include_groups=False: 避免 pandas 2.x 对分组列操作的 FutureWarning
    return df.groupby("symbol", group_keys=False).apply(_calc_group, include_groups=False).reset_index(level=0, drop=True)


def calc_cci_20(df: pd.DataFrame) -> pd.Series:
    """
    商品通道指标（CCI）
    TP = (high + low + close) / 3
    MA_TP = 20日TP的移动平均
    MD = 20日(|TP - MA_TP|)的平均值
    CCI = (TP - MA_TP) / (0.015 * MD)

    业务含义：CCI > 100 为超买，CCI < -100 为超卖
    A股特性：趋势行情中CCI在超买超卖区间停留时间较长
    """
    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] technical: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        tp = (group["high"] + group["low"] + group["close"]) / 3
        ma_tp = tp.rolling(20, min_periods=1).mean()
        # 平均绝对偏差
        md = (tp - ma_tp).abs().rolling(20, min_periods=1).mean()
        cci = (tp - ma_tp) / (0.015 * md + 1e-12)
        return cci

    # include_groups=False: 避免 pandas 2.x 对分组列操作的 FutureWarning
    return df.groupby("symbol", group_keys=False).apply(_calc_group, include_groups=False).reset_index(level=0, drop=True)


def calc_bb_bandwidth(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.Series:
    """
    布林带带宽 = (上轨 - 下轨) / 中轨
    带宽越宽表示波动率越大，带宽收窄往往预示重大变盘

    业务含义：低带宽通常预示着即将发生突破（无论是向上还是向下）
    A股特性：低带宽策略在A股震荡市中有较好表现
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ma = series.rolling(period, min_periods=1).mean()
        std = series.rolling(period, min_periods=1).std()
        upper = ma + k * std
        lower = ma - k * std
        bandwidth = (upper - lower) / (ma + 1e-12)
        return bandwidth

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_bb_position(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.Series:
    """
    价格在布林带中的相对位置 = (close - lower) / (upper - lower)
    0 = 触及下轨，1 = 触及上轨，0.5 = 中轨

    业务含义：位置 > 0.8 表示接近超买，位置 < 0.2 表示接近超卖
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ma = series.rolling(period, min_periods=1).mean()
        std = series.rolling(period, min_periods=1).std()
        upper = ma + k * std
        lower = ma - k * std
        position = (series - lower) / (upper - lower + 1e-12)
        return position

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_obv_chg_20(df: pd.DataFrame) -> pd.Series:
    """
    能量潮（OBV）20日变化率
    OBV = 累积(成交量 * sign(价格变化))
    OBV_CHG = (OBV_current - OBV_20_before) / OBV_20_before

    业务含义：OBV创新高而价格未创新高为背离信号（看涨）
    A股特性：OBV在A股中有一定的领先性，但不如量价背离直接
    """
    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] technical: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        # 计算每日OBV变动量
        close_chg = group["close"].diff()
        sign = np.sign(close_chg)
        # 价格持平当日不计入OBV
        obv_delta = sign * group["volume"]
        obv = obv_delta.cumsum()

        # 20日变化率
        obv_20_ago = obv.shift(20)
        chg = (obv - obv_20_ago) / (obv_20_ago.abs() + 1e-12)
        return chg

    # include_groups=False: 避免 pandas 2.x 对分组列操作的 FutureWarning
    return df.groupby("symbol", group_keys=False).apply(_calc_group, include_groups=False).reset_index(level=0, drop=True)


def _register_technical_factors():
    """注册技术指标因子（共11个）"""
    register(
        name="RSI_14",
        category="technical",
        func=lambda df: calc_rsi(df, 14),
        comment="14日相对强弱指标，>70超买，<30超卖"
    )
    register(
        name="RSI_21",
        category="technical",
        func=lambda df: calc_rsi(df, 21),
        comment="21日相对强弱指标，更长周期版本"
    )
    register(
        name="MACD_LINE",
        category="technical",
        func=calc_macd_line,
        comment="MACD线 = 12日EMA - 26日EMA，>0为多头趋势"
    )
    register(
        name="MACD_HIST",
        category="technical",
        func=calc_macd_hist,
        comment="MACD柱状图 = MACD线 - 信号线，上穿0轴为买入信号"
    )
    register(
        name="KDJ_J",
        category="technical",
        func=lambda df: calc_kdj_j(df, 9),
        comment="KDJ J值，>100超买，<0超卖"
    )
    register(
        name="CCI_20",
        category="technical",
        func=calc_cci_20,
        comment="20日商品通道指标，>100超买，<-100超卖"
    )
    register(
        name="BB_BANDWIDTH",
        category="technical",
        func=lambda df: calc_bb_bandwidth(df, 20, 2.0),
        comment="布林带带宽（20日，2倍标准差），低值预示变盘"
    )
    register(
        name="BB_POSITION",
        category="technical",
        func=lambda df: calc_bb_position(df, 20, 2.0),
        comment="价格在布林带中的相对位置（0~1），>0.8接近超买"
    )
    register(
        name="OBV_CHG_20",
        category="technical",
        func=calc_obv_chg_20,
        comment="能量潮20日变化率，捕捉量价背离信号"
    )

    # 威廉指标（14/28 日）——超卖时值高（反转看涨）
    for w in [14, 28]:
        register(
            name=f"WILLR_{w}",
            category="technical",
            func=lambda df, w=w: calc_willr(df, w),
            comment=f"{w}日威廉指标 = (HH{ w }-close)/(HH{ w }-LL{ w })，超卖=1（值高看涨）"
        )

    # 心理线（12/24 日）——上涨天数占比，情绪指标
    for w in [12, 24]:
        register(
            name=f"PSY_{w}",
            category="technical",
            func=lambda df, w=w: calc_psy(df, w),
            comment=f"{w}日心理线 = 上涨天数占比，>0.75超买，<0.25超卖"
        )

    # 三重指数平滑 TRIX（12/24）——过滤噪音的动量
    for w in [12, 24]:
        register(
            name=f"TRIX_{w}",
            category="technical",
            func=lambda df, w=w: calc_trix(df, w),
            comment=f"{w}日三重指数平滑变化率，趋势过滤版动量"
        )

    # EMA 快慢线交叉（5-20 / 10-30）——趋势强度
    for fast, slow in [(5, 20), (10, 30)]:
        register(
            name=f"EMA_CROSS_{fast}_{slow}",
            category="technical",
            func=lambda df, fast=fast, slow=slow: calc_ema_cross(df, fast, slow),
            comment=f"EMA{fast}-EMA{slow}差值相对价格，>0多头趋势"
        )


def calc_willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    威廉指标 %R（简化版）= (HH_N - close) / (HH_N - LL_N)，范围 [0, 1]。

    超买（close≈HH_N）时 ≈ 0；超卖（close≈LL_N）时 ≈ 1。
    A股反转逻辑：超卖（值高）→ 看涨，超买（值低）→ 看跌。
    """
    required = ["high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] technical: 缺少字段 {missing}")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        hh = group["high"].rolling(period, min_periods=1).max()
        ll = group["low"].rolling(period, min_periods=1).min()
        return (hh - group["close"]) / (hh - ll + 1e-12)

    return df.groupby("symbol", group_keys=False).apply(_calc_group, include_groups=False).reset_index(level=0, drop=True)


def calc_psy(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """
    心理线 PSY = 近 N 日上涨天数占比，范围 [0, 1]。

    PSY > 0.75 超买（乐观情绪过热），PSY < 0.25 超卖（悲观情绪过度）。
    与 UP_DOWN_RATIO20 类似，但周期更短、侧重情绪周期。
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        up = (series.pct_change() > 0).astype(float)
        return up.rolling(period, min_periods=1).mean()

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_trix(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """
    三重指数平滑 TRIX = EMA3(EMA3(EMA3(close))) 的变化率。

    对价格做三次指数平滑，消除短期噪音后测动量：
    值 > 0 表示三重平滑趋势向上，< 0 表示向下。
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.pct_change()

    return df.groupby("symbol")["close"].transform(_calc_group)


def calc_ema_cross(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """
    EMA 快慢线交叉 = (EMA_fast - EMA_slow) / close。

    除以价格去量纲。值 > 0 表示快线在上（多头排列），< 0 表示空头排列。
    与 MA_ALIGN_5_20（SMA）互补：EMA 对近期价格权重更高，响应更快。
    """
    if "close" not in df.columns:
        print("[WARN] technical: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow) / (series + 1e-12)

    return df.groupby("symbol")["close"].transform(_calc_group)


# 执行注册（模块导入时自动调用）
_register_technical_factors()