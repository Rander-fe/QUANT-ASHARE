# -*- coding: utf-8 -*-
"""
反转与动量因子（Reversal & Momentum）

实现因子：
    - REV5/10/20/60: 短期反转（过去 N 日收益率取负）
    - MOM20/60: N 日动量
    - MOM_SKIP_1M: 跳 1 月动量（skip-1-month，12-1）
    - MAX_DD20/60: 最大回撤（反转逻辑）
    - RISK_ADJ_MOM20/60: 风险调整动量（收益率/收益率波动率）
"""

import numpy as np
import pandas as pd

from factors.registry import register
from factors.operators import ts_pct_change


def _by_symbol(df: pd.DataFrame, col: str, func) -> pd.Series:
    """按股票分组计算，避免不同股票之间边界污染（pct_change/rolling 会跨组）。

    等价于 Alpha158 按 instrument 分组的时序算子，返回与 df 行对齐的序列。
    """
    return df.groupby("symbol")[col].transform(func)


def calc_rev(close: pd.Series, window: int) -> pd.Series:
    """N 日反转因子（收益率取负，越高代表过去跌得越多）"""
    ret = ts_pct_change(close, window)
    return -ret


def calc_mom(close: pd.Series, window: int) -> pd.Series:
    """N 日动量因子（直接使用收益率）"""
    return ts_pct_change(close, window)


def calc_mom_skip_1m(close: pd.Series) -> pd.Series:
    """跳 1 月动量（skip-1-month momentum，12-1 动量）。

    跳过最近一个月（避免短期反转干扰），取过去约一年减去最近一月的动量：
        close.shift(21) / close.shift(252) - 1
    出处：Jegadeesh & Titman (1993) 的 12-1 动量，避免最近一个月反转抵消动量。
    """
    return close.shift(21) / (close.shift(252) + 1e-12) - 1


def calc_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    """
    最大回撤因子（过去 window 日内振幅回撤幅度）。

    回撤越大因子值越大（反转逻辑：超跌越深越可能反弹，值越大越看好）。
    方向约定：值越大 = 预期收益越高（与注册表方向一致）。
    """
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(result, index=close.index)
    # 每只股票通常只有数千行；矩阵化比逐窗口Python回调快很多。
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    running_peaks = np.maximum.accumulate(windows, axis=1)
    drawdowns = windows / running_peaks - 1.0
    result[window - 1:] = -np.min(drawdowns, axis=1)
    return pd.Series(result, index=close.index)


def calc_risk_adj_mom(close: pd.Series, window: int) -> pd.Series:
    """
    风险调整动量 = 收益率 / 收益率波动率（Sharpe 式，无价格量纲）。

    分母用「日收益率的滚动标准差」，而非价格标准差：
        ret(window) / std(daily_ret, window)
    去量纲：收益率除以收益率波动率，消除价格水平差异。
    """
    ret = ts_pct_change(close, window)
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(window, min_periods=window).std()
    return ret / (vol + 1e-12)


def calc_intraday_rev(df: pd.DataFrame, window: int) -> pd.Series:
    """
    日内反转因子 = -mean(close/open - 1, N)。

    衡量「日内走势强度」：收在开盘价之上为强（值为负），收在其下为弱（值为正）。
    取负后：日内越弱（尾盘杀跌）因子值越高 → A股T+1制度下，
    当日被套的资金次日倾向割肉/反弹，日内弱势股次日常有均值回归。
    """
    if "open" not in df.columns or "close" not in df.columns:
        print("[WARN] reversal_momentum: 缺少 open/close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        intra = group["close"] / (group["open"] + 1e-12) - 1
        return -intra.rolling(window, min_periods=1).mean()

    return (
        df.groupby("symbol", group_keys=False)
        .apply(_calc_group, include_groups=False)
        .reset_index(level=0, drop=True)
    )


def calc_overnight_rev(df: pd.DataFrame, window: int) -> pd.Series:
    """
    隔夜反转因子 = -mean(open / prev_close - 1, N)。

    衡量隔夜跳空方向：高开（值为正）说明隔夜情绪乐观，低开（值为负）说明悲观。
    取负后：连续低开（隔夜弱势）因子值越高 → 隔夜反应过度的股票次日倾向反转。
    """
    if "open" not in df.columns or "close" not in df.columns:
        print("[WARN] reversal_momentum: 缺少 open/close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(group: pd.DataFrame) -> pd.Series:
        prev_close = group["close"].shift(1)
        overnight = group["open"] / (prev_close + 1e-12) - 1
        return -overnight.rolling(window, min_periods=1).mean()

    return (
        df.groupby("symbol", group_keys=False)
        .apply(_calc_group, include_groups=False)
        .reset_index(level=0, drop=True)
    )


def calc_consec_up(df: pd.DataFrame, window: int) -> pd.Series:
    """
    连续同向K线计数 = 最近 window 日内连续收阳的天数。

    用符号序列 sign(ret) 的滚动窗口「同号连续计数」实现：
        从当前日往前数，连续 sign 相同（均为正）的天数，上限为 window。
    值越高 = 连续上涨天数越多（超买风险积累）→ 取负后值越高代表
    连续上涨后回调概率增大（反转逻辑）。
    """
    if "close" not in df.columns:
        print("[WARN] reversal_momentum: 缺少 close 字段")
        return pd.Series(index=df.index, dtype=float)

    def _calc_group(series: pd.Series) -> pd.Series:
        ret = series.pct_change(fill_method=None)
        up = (ret > 0).astype(float).where(ret.notna())

        def _trailing_up(values: np.ndarray) -> float:
            count = 0
            for value in values[::-1]:
                if value != 1.0:
                    break
                count += 1
            return float(count)

        return up.rolling(window, min_periods=window).apply(_trailing_up, raw=True)

    up_count = (
        df.groupby("symbol")["close"]
        .transform(lambda s: _calc_group(s))
    )
    return -up_count


def calc_mom_skip_1w(close: pd.Series) -> pd.Series:
    """跳 1 周动量（skip-1-week，20-5）动量。

    跳过最近 5 日（避免极短期反转干扰），取 20 日动量减去最近 5 日：
        close.shift(5) / close.shift(25) - 1
    与 MOM_SKIP_1M 互补：一个过滤月内噪音，一个过滤周内噪音。
    """
    return close.shift(5) / (close.shift(25) + 1e-12) - 1


def calc_up_down_ratio(close: pd.Series, window: int) -> pd.Series:
    """
    上涨天数占比 = mean(sign(ret) > 0, N)，范围 [0, 1]。

    值 > 0.5：多数时间在涨（买方主导）；< 0.5：多数时间在跌（卖方主导）。
    与收益率不同：弱化了极端单日涨跌的权重，更稳健地刻画「上涨广度」。
    """
    daily_ret = close.pct_change()
    up = (daily_ret > 0).astype(float)
    return up.rolling(window, min_periods=1).mean()


# ============================================================
# 注册所有因子到注册表
# ============================================================

def _register_reversal_momentum_factors():
    """注册反转/动量因子（延迟注册，确保函数已定义）"""

    # 反转因子（5/10/20/60 日）
    for w in [5, 10, 20, 60]:
        register(
            name=f"REV{w}",
            category="reversal_momentum",
            func=lambda df, w=w: _by_symbol(df, "close", lambda s: calc_rev(s, w)),
            comment=f"{w}日反转因子，越高表示过去{w}日跌幅越大（A股最强Alpha）"
        )

    # 动量因子（20/60 日）
    for w in [20, 60]:
        register(
            name=f"MOM{w}",
            category="reversal_momentum",
            func=lambda df, w=w: _by_symbol(df, "close", lambda s: calc_mom(s, w)),
            comment=f"{w}日动量因子（A股动量较弱，建议作为辅助）"
        )

    # 跳 1 月动量（skip-1-month，12-1）
    register(
        name="MOM_SKIP_1M",
        category="reversal_momentum",
        func=lambda df: _by_symbol(df, "close", calc_mom_skip_1m),
        comment="跳1月动量 = 跳过最近一个月的一年动量（12-1，Jegadeesh&Titman 1993）"
    )

    # 最大回撤（20/60 日）
    for w in [20, 60]:
        register(
            name=f"MAX_DD{w}",
            category="reversal_momentum",
            func=lambda df, w=w: _by_symbol(df, "close", lambda s: calc_max_drawdown(s, w)),
            comment=f"{w}日最大回撤因子，回撤越大值越大（反转逻辑，值越大越看好）"
        )

    # 风险调整动量（20/60 日）
    for w in [20, 60]:
        register(
            name=f"RISK_ADJ_MOM{w}",
            category="reversal_momentum",
            func=lambda df, w=w: _by_symbol(df, "close", lambda s: calc_risk_adj_mom(s, w)),
            comment=f"{w}日风险调整动量 = 收益率/收益率波动率（Sharpe式，无价格量纲）"
        )

    # 日内反转（5/10 日）——T+1 制度下日内弱势股次日均值回归
    for w in [5, 10]:
        register(
            name=f"INTRADAY_REV{w}",
            category="reversal_momentum",
            func=lambda df, w=w: calc_intraday_rev(df, w),
            comment=f"{w}日日内反转 = -mean(close/open-1)，日内越弱因子值越高（T+1均值回归）"
        )

    # 隔夜反转（5/10 日）——隔夜跳空过度反应的均值回归
    for w in [5, 10]:
        register(
            name=f"OVERNIGHT_RET{w}",
            category="reversal_momentum",
            func=lambda df, w=w: calc_overnight_rev(df, w),
            comment=f"{w}日隔夜反转 = -mean(open/prev_close-1)，连续低开因子值越高"
        )

    # 连续上涨K线计数（3/5 日）——超买回调风险
    for w in [3, 5]:
        register(
            name=f"CONSEC_UP{w}",
            category="reversal_momentum",
            func=lambda df, w=w: calc_consec_up(df, w),
            comment=f"近{w}日连续上涨天数取负，连涨越多超买回调概率越大（反转）"
        )

    # 跳 1 周动量（20-5）
    register(
        name="MOM_SKIP_1W",
        category="reversal_momentum",
        func=lambda df: _by_symbol(df, "close", calc_mom_skip_1w),
        comment="跳1周动量 = 跳过最近5日取20日动量（20-5），与MOM_SKIP_1M互补"
    )

    # 上涨天数占比（20 日）
    register(
        name="UP_DOWN_RATIO20",
        category="reversal_momentum",
        func=lambda df: _by_symbol(df, "close", lambda s: calc_up_down_ratio(s, 20)),
        comment="20日上涨天数占比，弱化极端单日影响，刻画上涨广度"
    )


# 执行注册
_register_reversal_momentum_factors()
