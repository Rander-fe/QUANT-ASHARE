# -*- coding: utf-8 -*-
"""
成长因子（Growth）

实现因子：
    - REV_YOY: 营收同比增速（来自 fina_indicator.revenue_yoy）
    - PROFIT_YOY: 净利润同比增速（来自 fina_indicator.netprofit_yoy）
    - REV_ACCEL: 营收增速加速度（营收同比增速的季度变化）
    - PROFIT_ACCEL: 净利润增速加速度（净利润同比增速的季度变化）

数据说明：
    - 财务字段已通过 ann_date 对齐，并合并到底表。
    - 每个财报日（季末）才有值，期间前向填充后仍可能为 NaN。
    - 加速度计算：按股票分组，对同比增速序列做差分（diff）。
"""

import pandas as pd

from factors.registry import register


def _get_rev_yoy_col(df: pd.DataFrame) -> object:
    """返回可用的营收同比列名：优先 tr_yoy（营业总收入同比），其次 or_yoy（营收同比），
    最后兼容旧数据 revenue_yoy。三者均为 Tushare fina_indicator 的营收同比增速字段。"""
    for col in ("tr_yoy", "or_yoy", "revenue_yoy"):
        if col in df.columns:
            return col
    return None


def calc_rev_yoy(df: pd.DataFrame) -> pd.Series:
    """
    营收同比增速（直接取字段）
    """
    col = _get_rev_yoy_col(df)
    if col is None:
        print("[WARN] growth: 缺少 tr_yoy/or_yoy/revenue_yoy 字段，请确认底表已合并财务数据")
        return pd.Series(index=df.index, dtype=float)
    # 原字段可能已经是小数形式（如 0.20），也可能是百分比（20），为保险统一除以 100
    # 但更常见的 Tushare 返回的是百分比数值，如 20.5，除以 100 转为小数。
    # 这里不做自动转换，因为因子值本身只用于相对排序，数值量纲不重要。
    # 若需标准化，后续由 Qlib DataHandler 处理。
    return df[col].astype(float)


def calc_profit_yoy(df: pd.DataFrame) -> pd.Series:
    """
    净利润同比增速（直接取字段）
    """
    if "netprofit_yoy" not in df.columns:
        print("[WARN] growth: 缺少 netprofit_yoy 字段")
        return pd.Series(index=df.index, dtype=float)
    return df["netprofit_yoy"].astype(float)


def calc_rev_accel(df: pd.DataFrame) -> pd.Series:
    """
    营收增速加速度 = 营收同比增速的季度环比变化（即当前季度增速 - 上一季度增速）
    按股票分组，对营收同比序列做一阶差分。
    """
    col = _get_rev_yoy_col(df)
    if col is None:
        print("[WARN] growth: 缺少 tr_yoy/or_yoy/revenue_yoy，无法计算营收加速度")
        return pd.Series(index=df.index, dtype=float)

    # 确保按日期排序（但 df 可能已全局排序，分组内也需保持顺序）
    # 为避免意外，在分组内排序
    grouped = df.groupby("symbol", group_keys=False)
    # 注意：需要保证每个分组内按日期升序，由于 df 在外部已按 [symbol, date] 排序，这里可省略
    return grouped[col].transform(lambda x: x.diff())


def calc_profit_accel(df: pd.DataFrame) -> pd.Series:
    """
    净利润增速加速度 = 净利润同比增速的季度环比变化
    """
    if "netprofit_yoy" not in df.columns:
        print("[WARN] growth: 缺少 netprofit_yoy，无法计算净利润加速度")
        return pd.Series(index=df.index, dtype=float)

    grouped = df.groupby("symbol", group_keys=False)
    return grouped["netprofit_yoy"].transform(lambda x: x.diff())


# ============================================================
# 扩展成长因子（依赖全量 fina_indicator 108 列）
# ============================================================
def _direct_field(df: pd.DataFrame, field: str, module: str = "growth") -> pd.Series:
    """直接取 fina_indicator 字段（字段缺失时优雅降级为空序列）"""
    if field not in df.columns:
        print(f"[WARN] {module}: 缺少 {field} 字段")
        return pd.Series(index=df.index, dtype=float)
    return df[field].astype(float)


def calc_op_yoy(df: pd.DataFrame) -> pd.Series:
    """营业利润同比增速（op_yoy）：剔除投资收益等非主业噪音后的主业增速，
    比净利润同比更干净。"""
    return _direct_field(df, "op_yoy")


def calc_ocf_yoy(df: pd.DataFrame) -> pd.Series:
    """经营现金流同比增速（ocf_yoy）：现金流维度的成长，
    与利润增速背离时需警惕应收堆积。"""
    return _direct_field(df, "ocf_yoy")


def calc_roe_yoy(df: pd.DataFrame) -> pd.Series:
    """ROE 同比变化（roe_yoy）：盈利能力改善的方向与幅度，
    正值为盈利能力改善，比绝对 ROE 更具边际信息。"""
    return _direct_field(df, "roe_yoy")


def calc_q_sales_yoy(df: pd.DataFrame) -> pd.Series:
    """单季营收同比（q_sales_yoy）：剔除累计口径的季节性平滑，
    更及时捕捉最新季度的景气拐点。"""
    return _direct_field(df, "q_sales_yoy")


def calc_q_op_qoq(df: pd.DataFrame) -> pd.Series:
    """单季营业利润环比（q_op_qoq）：环比视角的主业动能，
    环比为正代表季度间持续改善。"""
    return _direct_field(df, "q_op_qoq")


def _register_growth_factors():
    """注册成长因子"""
    register(
        name="REV_YOY",
        category="growth",
        func=calc_rev_yoy,
        comment="营收同比增速，来自合并财报，数值为百分比（如20.5表示20.5%）"
    )
    register(
        name="PROFIT_YOY",
        category="growth",
        func=calc_profit_yoy,
        comment="净利润同比增速，来自合并财报"
    )
    register(
        name="REV_ACCEL",
        category="growth",
        func=calc_rev_accel,
        comment="营收增速加速度 = 本期营收同比增速 - 上期营收同比增速"
    )
    register(
        name="PROFIT_ACCEL",
        category="growth",
        func=calc_profit_accel,
        comment="净利润增速加速度 = 本期净利润同比增速 - 上期净利润同比增速"
    )
    # ---- 扩展（全量 fina_indicator 字段）----
    register(
        name="OP_YOY",
        category="growth",
        func=calc_op_yoy,
        comment="营业利润同比增速，op_yoy，主业增速更干净"
    )
    register(
        name="OCF_YOY",
        category="growth",
        func=calc_ocf_yoy,
        comment="经营现金流同比增速，ocf_yoy，现金流成长维度"
    )
    register(
        name="ROE_YOY",
        category="growth",
        func=calc_roe_yoy,
        comment="ROE同比变化，roe_yoy，盈利改善的边际信息"
    )
    register(
        name="Q_SALES_YOY",
        category="growth",
        func=calc_q_sales_yoy,
        comment="单季营收同比，q_sales_yoy，季度景气拐点最快信号"
    )
    register(
        name="Q_OP_QOQ",
        category="growth",
        func=calc_q_op_qoq,
        comment="单季营业利润环比，q_op_qoq，主业季度动能"
    )


# 执行注册（模块导入时自动调用）
_register_growth_factors()