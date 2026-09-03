# -*- coding: utf-8 -*-
"""
估值因子（Value）

实现因子（共3个）：
    - EP = 1 / pe_ttm（市盈率倒数）：值越高，代表估值越低（越便宜）
    - BP = 1 / pb（市净率倒数）：值越高，代表市净率越低（越便宜）
    - SP = 1 / ps_ttm（市销率倒数）：值越高，代表市销率越低（越便宜）

数据来源：
    - pe_ttm / pb / ps_ttm 来自 daily_basic（tushare），已合并到底表

A股估值因子特性：
    - BP（市净率倒数）在A股中表现优于EP，长期IC更稳定
    - EP在盈利稳定行业中有效（如银行、消费）
    - SP对成长型公司（尚未盈利）更具参考价值

⚠️ 关键防御：
    - pe_ttm / pb / ps_ttm 可能为 0 或负值（亏损公司）
    - 直接取倒数会导致 inf 或 -inf，必须做安全处理
    - 处理规则：分母为 0 或负值 → 返回 0（表示中性/无意义）
"""

import numpy as np
import pandas as pd

from factors.registry import register


def _safe_inverse(series: pd.Series) -> pd.Series:
    """
    安全取倒数：
        - 分母为 0 → 返回 0（而非 inf）
        - 分母为负 → 返回 0（负估值无意义）
        - 分母为正 → 返回 1 / value
    """
    # 先转为 float，处理 NaN
    s = series.astype(float)
    # 只对 > 0 的值取倒数，其余（<=0 或 NaN）返回 0
    # 注意：np.where 会保持 dtype，需要显式指定
    result = np.where(s > 0, 1.0 / s, 0.0)
    return pd.Series(result, index=series.index)


def calc_ep(df: pd.DataFrame) -> pd.Series:
    """
    EP = 1 / pe_ttm（市盈率倒数）
    - pe_ttm > 0 才有经济含义（盈利为正）
    - pe_ttm = 0 或为负 → 返回 0（不参与排序）
    """
    if "pe_ttm" not in df.columns:
        print("[WARN] value: 缺少 pe_ttm 字段")
        return pd.Series(index=df.index, dtype=float)
    return _safe_inverse(df["pe_ttm"])


def calc_bp(df: pd.DataFrame) -> pd.Series:
    """
    BP = 1 / pb（市净率倒数）
    - pb > 0 才有经济含义（净资产为正）
    - pb = 0 或为负 → 返回 0（不参与排序）
    - A股中 BP 因子的 IC 通常高于 EP
    """
    if "pb" not in df.columns:
        print("[WARN] value: 缺少 pb 字段")
        return pd.Series(index=df.index, dtype=float)
    return _safe_inverse(df["pb"])


def calc_sp(df: pd.DataFrame) -> pd.Series:
    """
    SP = 1 / ps_ttm（市销率倒数）
    - ps_ttm > 0 才有经济含义（营收为正）
    - ps_ttm = 0 或为负 → 返回 0（不参与排序）
    - 对尚未盈利的成长型公司，SP 比 EP 更有参考价值
    """
    if "ps_ttm" not in df.columns:
        print("[WARN] value: 缺少 ps_ttm 字段")
        return pd.Series(index=df.index, dtype=float)
    return _safe_inverse(df["ps_ttm"])


def _register_value_factors():
    """注册估值因子（共3个）"""
    register(
        name="EP",
        category="value",
        func=calc_ep,
        comment="市盈率倒数 EP = 1/pe_ttm，值越高越便宜，仅 pe_ttm > 0 时有效"
    )
    register(
        name="BP",
        category="value",
        func=calc_bp,
        comment="市净率倒数 BP = 1/pb，值越高越便宜，A股最强价值因子之一"
    )
    register(
        name="SP",
        category="value",
        func=calc_sp,
        comment="市销率倒数 SP = 1/ps_ttm，值越高越便宜，适用于成长型公司"
    )


# 执行注册（模块导入时自动调用）
_register_value_factors()