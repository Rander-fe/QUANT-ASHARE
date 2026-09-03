# -*- coding: utf-8 -*-
"""
质量因子（Quality）

实现因子（共10个）：
    - ROE: 净资产收益率（直接取值），衡量股东权益的回报效率
    - ROA: 总资产收益率（直接取值），衡量全部资产的回报效率
    - GPM: 毛利率（grossprofit_margin），产品竞争力的核心指标
    - NPM: 净利率（netprofit_margin），盈利能力的最终体现
    - CURRENT_RATIO: 流动比率（current_ratio），短期偿债能力
    - LOW_DEBT: 低负债因子（-debt_to_assets），资产负债率取负，越低越好
    - ROE_WAA: 加权净资产收益率（roe_waa），机构常用
    - ROIC: 投入资本回报率（roic），剔除杠杆的真实资本回报
    - QUICK_RATIO: 速动比率（quick_ratio），更严格的偿债指标
    - ASSETS_TURN: 总资产周转率（assets_turn），资产运营效率

数据来源：
    - 所有字段来自 fina_indicator（tushare），已通过 ann_date 对齐
    - 合并到底表 basic_cleaned_with_extra.parquet 中

A股质量因子特性：
    - ROE/ROA 在A股中长期有效，尤其适用于长期持有策略
    - 毛利率比净利率更稳定，抗操纵性更强
    - 低负债在熊市中防御性显著
"""

import pandas as pd

from factors.registry import register


def calc_roe(df: pd.DataFrame) -> pd.Series:
    """
    净资产收益率（ROE）= 净利润 / 净资产
    直接取 fina_indicator.roe 字段值
    """
    if "roe" not in df.columns:
        print("[WARN] quality: 缺少 roe 字段")
        return pd.Series(index=df.index, dtype=float)
    return df["roe"].astype(float)


def calc_roa(df: pd.DataFrame) -> pd.Series:
    """
    总资产收益率（ROA）= 净利润 / 总资产
    直接取 fina_indicator.roa 字段值
    """
    if "roa" not in df.columns:
        print("[WARN] quality: 缺少 roa 字段")
        return pd.Series(index=df.index, dtype=float)
    return df["roa"].astype(float)


def calc_gpm(df: pd.DataFrame) -> pd.Series:
    """
    毛利率 = (营业收入 - 营业成本) / 营业收入
    直接取 fina_indicator.grossprofit_margin 字段值
    毛利率代表产品的核心竞争力，受操纵影响小，比净利率更可靠
    """
    if "grossprofit_margin" not in df.columns:
        print("[WARN] quality: 缺少 grossprofit_margin 字段")
        return pd.Series(index=df.index, dtype=float)
    return df["grossprofit_margin"].astype(float)


def calc_npm(df: pd.DataFrame) -> pd.Series:
    """
    净利率 = 净利润 / 营业收入
    直接取 fina_indicator.netprofit_margin 字段值
    净利率代表公司整体盈利效率，容易受非经常性损益影响
    """
    if "netprofit_margin" not in df.columns:
        print("[WARN] quality: 缺少 netprofit_margin 字段")
        return pd.Series(index=df.index, dtype=float)
    return df["netprofit_margin"].astype(float)


def calc_current_ratio(df: pd.DataFrame) -> pd.Series:
    """
    流动比率 = 流动资产 / 流动负债
    直接取 fina_indicator.current_ratio 字段值
    衡量企业短期偿债能力，值越高代表短期财务越安全
    """
    if "current_ratio" not in df.columns:
        print("[WARN] quality: 缺少 current_ratio 字段")
        return pd.Series(index=df.index, dtype=float)
    return df["current_ratio"].astype(float)


def calc_low_debt(df: pd.DataFrame) -> pd.Series:
    """
    低负债因子 = -debt_to_assets（资产负债率取负）
    资产负债率 = 总负债 / 总资产
    负债率越高，公司财务风险越大；取负后，值越高代表负债率越低（越安全）
    """
    if "debt_to_assets" not in df.columns:
        print("[WARN] quality: 缺少 debt_to_assets 字段")
        return pd.Series(index=df.index, dtype=float)
    return -df["debt_to_assets"].astype(float)


# ============================================================
# 扩展质量因子（依赖全量 fina_indicator 108 列）
# ============================================================
def _direct_field(df: pd.DataFrame, field: str, warn: bool = True) -> pd.Series:
    """直接取 fina_indicator 字段（字段缺失时优雅降级为空序列）"""
    if field not in df.columns:
        if warn:
            print(f"[WARN] quality: 缺少 {field} 字段")
        return pd.Series(index=df.index, dtype=float)
    return df[field].astype(float)


def calc_roe_waa(df: pd.DataFrame) -> pd.Series:
    """加权净资产收益率（roe_waa）：比摊薄 ROE 更能反映期间加权资本回报，
    机构常用，A股质量因子的核心候选。"""
    return _direct_field(df, "roe_waa")


def calc_roic(df: pd.DataFrame) -> pd.Series:
    """投入资本回报率（roic）：EBIT(1-税率)/投入资本，衡量全部投入资本的
    真实回报，剔除财务杠杆影响，跨行业可比性更好。"""
    return _direct_field(df, "roic")


def calc_quick_ratio(df: pd.DataFrame) -> pd.Series:
    """速动比率（quick_ratio）：(流动资产-存货)/流动负债，比流动比率更严格
    的短期偿债指标，剔除难以快速变现的存货。"""
    return _direct_field(df, "quick_ratio")


def calc_assets_turn(df: pd.DataFrame) -> pd.Series:
    """总资产周转率（assets_turn）：营业收入/平均总资产，衡量资产运营效率，
    周转率高的公司轻资产运营能力强。"""
    return _direct_field(df, "assets_turn")


def _register_quality_factors():
    """注册质量因子（共10个）"""
    register(
        name="ROE",
        category="quality",
        func=calc_roe,
        comment="净资产收益率，直接取 roe 字段，越高越优质"
    )
    register(
        name="ROA",
        category="quality",
        func=calc_roa,
        comment="总资产收益率，直接取 roa 字段，越高越优质"
    )
    register(
        name="GPM",
        category="quality",
        func=calc_gpm,
        comment="毛利率，grossprofit_margin，产品竞争力核心指标"
    )
    register(
        name="NPM",
        category="quality",
        func=calc_npm,
        comment="净利率，netprofit_margin，盈利效率最终体现"
    )
    register(
        name="CURRENT_RATIO",
        category="quality",
        func=calc_current_ratio,
        comment="流动比率，current_ratio，短期偿债能力"
    )
    register(
        name="LOW_DEBT",
        category="quality",
        func=calc_low_debt,
        comment="低负债因子，-debt_to_assets，值越高负债率越低，财务越安全"
    )
    # ---- 扩展（全量 fina_indicator 字段）----
    register(
        name="ROE_WAA",
        category="quality",
        func=calc_roe_waa,
        comment="加权净资产收益率，roe_waa，机构常用的期间加权ROE"
    )
    register(
        name="ROIC",
        category="quality",
        func=calc_roic,
        comment="投入资本回报率，roic，剔除杠杆的真实资本回报"
    )
    register(
        name="QUICK_RATIO",
        category="quality",
        func=calc_quick_ratio,
        comment="速动比率，quick_ratio，比流动比率更严格的偿债指标"
    )
    register(
        name="ASSETS_TURN",
        category="quality",
        func=calc_assets_turn,
        comment="总资产周转率，assets_turn，资产运营效率"
    )


# 执行注册（模块导入时自动调用）
_register_quality_factors()