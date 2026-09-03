# -*- coding: utf-8 -*-
"""
因子进阶校验：10分组单调性、换手率、因子相关性（冗余剔除）

输入：
    - data/processed/factors.parquet（因子 + 标签）
    - data/processed/basic_cleaned_with_extra_by_date.parquet（含 turnover_rate 和 close）
输出：
    - data/processed/factor_monotonicity.parquet（分组收益表）
    - data/processed/factor_turnover.parquet（换手率表）
    - data/processed/factor_redundancy.parquet（冗余对清单）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED


def check_monotonicity(
    df_factors: pd.DataFrame,
    df_base: pd.DataFrame,
    factor_cols: list,
    label_col: str = "label_ret_5",
) -> pd.DataFrame:
    """
    对每个因子做 10 分组单调性检验（按日分组，取 10 等分）。
    返回：因子名、D1~D10 平均收益、多空收益（Top - Bottom）、是否单调递增。
    """
    # 合并标签和因子数据（确保日期和股票对齐）
    df_merged = pd.merge(
        df_factors[["symbol", "date", label_col] + factor_cols],
        df_base[["symbol", "date"]],  # 仅保留基础数据中的有效交易日
        on=["symbol", "date"],
        how="inner",
    )
    
    rows = []
    for factor in factor_cols:
        # 按日期分组
        daily_groups = df_merged.groupby("date")
        
        # 存储每个日期下各组的多空收益
        spread_list = []
        group_means = {f"D{i+1}": [] for i in range(10)}
        
        for _, group in daily_groups:
            if group[factor].isna().sum() > 0.5 * len(group):
                continue  # 当日因子缺失过多则跳过
            
            # 按因子值分成 10 组（用 rank + qcut 方法，避免 qcut 因重复值报错）
            group["rank_pct"] = group[factor].rank(pct=True)
            group["group"] = pd.cut(
                group["rank_pct"], bins=10, labels=[f"D{i+1}" for i in range(10)]
            )
            
            # 计算各组平均收益
            means = group.groupby("group")[label_col].mean()
            
            # 填充缺失组
            for g in [f"D{i+1}" for i in range(10)]:
                if g in means.index:
                    group_means[g].append(means[g])
                else:
                    group_means[g].append(np.nan)
            
            # 多空收益（D1 - D10，假设 D1 是因子值最高的组）
            if "D1" in means.index and "D10" in means.index:
                spread_list.append(means["D1"] - means["D10"])
        
        # 计算各组的平均收益（时间序列平均）
        row = {"factor": factor, "long_short_mean": np.nanmean(spread_list)}
        
        # 判断单调性：D1 > D2 > ... > D10（严格单调）
        d1_mean = np.nanmean(group_means["D1"])
        d10_mean = np.nanmean(group_means["D10"])
        
        # 如果 D1 明显大于 D10，且中间序列整体呈下降趋势，则视为有效
        row["d1_d10_diff"] = d1_mean - d10_mean
        
        # 简化判断：若 D1 > D10 且序列中位数符合单调趋势（使用 Spearman 检验）
        if d1_mean > d10_mean:
            seq_means = [np.nanmean(group_means[f"D{i+1}"]) for i in range(10)]
            # 单调性检验：序列是否单调递减
            # 计算相邻差是否大部分为负（D1 > D2 > ...）
            diff_sign = np.sign(np.diff(seq_means))
            monotonic_ratio = (diff_sign < 0).sum() / len(diff_sign)
            row["is_monotonic"] = monotonic_ratio >= 0.6  # 至少 60% 的相邻组符合递减
        else:
            row["is_monotonic"] = False
        
        rows.append(row)
    
    df_mono = pd.DataFrame(rows)
    return df_mono


def check_turnover(
    df_factors: pd.DataFrame,
    df_base: pd.DataFrame,
    factor_cols: list,
    turnover_col: str = "turnover_rate",
) -> pd.DataFrame:
    """
    计算 Top/Bottom 分组在当日的平均换手率（横截面均值）。
    换手率过高（如 > 30%）意味着该因子在实盘中会被高昂的交易成本吞噬。
    """
    # 合并因子与换手率数据
    df_merged = pd.merge(
        df_factors[["symbol", "date"] + factor_cols],
        df_base[["symbol", "date", turnover_col]],
        on=["symbol", "date"],
        how="inner",
    )
    
    rows = []
    for factor in factor_cols:
        # 按日分组，计算当日 Top 组（D1）和 Bottom 组（D10）的平均换手率
        top_turnovers = []
        bottom_turnovers = []
        
        for date, group in df_merged.groupby("date"):
            if group[factor].isna().sum() > 0.5 * len(group):
                continue
            
            group["rank_pct"] = group[factor].rank(pct=True)
            # 取前 10%（Top）和后 10%（Bottom）
            top_mask = group["rank_pct"] >= 0.9
            bottom_mask = group["rank_pct"] <= 0.1
            
            if top_mask.sum() > 0:
                top_turnovers.append(group.loc[top_mask, turnover_col].mean())
            if bottom_mask.sum() > 0:
                bottom_turnovers.append(group.loc[bottom_mask, turnover_col].mean())
        
        rows.append({
            "factor": factor,
            "top_turnover_mean": np.nanmean(top_turnovers) if top_turnovers else np.nan,
            "bottom_turnover_mean": np.nanmean(bottom_turnovers) if bottom_turnovers else np.nan,
            "turnover_over_30": np.nanmean(top_turnovers) > 0.30 if top_turnovers else False,
        })
    
    return pd.DataFrame(rows)


def check_factor_correlation(
    df_factors: pd.DataFrame,
    factor_cols: list,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """
    计算因子两两之间的相关系数（Spearman），标记高度相关的冗余对。
    若相关性 > threshold，建议剔除其中一个（通常保留 IC 更高的那个）。
    """
    # 先做横截面合并：将全部数据压缩成横截面，计算全局秩相关
    # 注意：这种方法会混合不同时间段的数据，但在因子筛选中是常用简化手段。
    # 更严谨的做法是按日计算相关系数后取平均（此处简化）
    
    df_sub = df_factors[factor_cols].dropna(how="all")
    if df_sub.empty:
        return pd.DataFrame(columns=["factor_1", "factor_2", "correlation"])
    
    # 计算 Spearman 相关系数矩阵（内存消耗较大，若因子 > 500 个需分块）
    corr_mat = df_sub.corr(method="spearman")
    
    # 提取上三角矩阵的冗余对
    redundant_pairs = []
    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            f1 = factor_cols[i]
            f2 = factor_cols[j]
            corr_val = corr_mat.loc[f1, f2]
            if abs(corr_val) > threshold:
                redundant_pairs.append({
                    "factor_1": f1,
                    "factor_2": f2,
                    "correlation": corr_val,
                })
    
    df_redundant = pd.DataFrame(redundant_pairs)
    return df_redundant


def main():
    # 1. 加载数据
    factor_path = DATA_PROCESSED / "factors.parquet"
    base_path = DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet"
    
    if not factor_path.exists():
        print(f"[ERROR] 未找到 {factor_path}，请先运行 build_factors.py")
        return 1
    
    if not base_path.exists():
        print(f"[ERROR] 未找到 {base_path}，请先运行 fetch_daily_basic.py")
        return 1
    
    df_factors = pd.read_parquet(factor_path)
    df_base = pd.read_parquet(base_path)
    print(f"📊 加载因子数据: {len(df_factors):,} 行")
    print(f"📊 加载基础数据: {len(df_base):,} 行")
    
    # 识别因子列
    exclude_cols = {"symbol", "date", "label_ret_5"}
    factor_cols = [c for c in df_factors.columns if c not in exclude_cols and c in df_base.columns]
    print(f"📊 待校验因子数: {len(factor_cols)}")
    
    # 2. 执行校验
    print("\n🔍 检查 10 分组单调性...")
    df_mono = check_monotonicity(df_factors, df_base, factor_cols)
    print(f"   ✅ 完成，有效因子: {df_mono['is_monotonic'].sum()} 个")
    
    print("\n🔄 检查因子换手率...")
    df_turnover = check_turnover(df_factors, df_base, factor_cols)
    print(f"   ✅ 完成，高换手因子: {df_turnover['turnover_over_30'].sum()} 个")
    
    print("\n📊 检查因子相关性...")
    df_redundant = check_factor_correlation(df_factors, factor_cols, threshold=0.7)
    print(f"   ✅ 完成，冗余对: {len(df_redundant)} 组")
    
    # 3. 保存结果
    mono_path = DATA_PROCESSED / "factor_monotonicity.parquet"
    df_mono.to_parquet(mono_path, index=False)
    print(f"✅ 单调性结果保存至: {mono_path}")
    
    turn_path = DATA_PROCESSED / "factor_turnover.parquet"
    df_turnover.to_parquet(turn_path, index=False)
    print(f"✅ 换手率结果保存至: {turn_path}")
    
    red_path = DATA_PROCESSED / "factor_redundancy.parquet"
    df_redundant.to_parquet(red_path, index=False)
    print(f"✅ 冗余因子对保存至: {red_path}")
    
    # 4. 打印摘要
    print("\n" + "=" * 80)
    print("📋 因子校验摘要")
    print("=" * 80)
    
    # 单调性：筛选出真正有效且单调的因子
    good_mono = df_mono[df_mono["is_monotonic"] & (df_mono["long_short_mean"] > 0)]
    print(f"✅ 单调性 + 多空收益为正的因子: {len(good_mono)} 个")
    
    # 换手率警告
    high_turn = df_turnover[df_turnover["turnover_over_30"]]
    if not high_turn.empty:
        print(f"⚠️ 换手率 > 30% 的因子（交易成本敏感）: {len(high_turn)} 个")
        print("   (建议在模型中使用前做稳健性处理，或直接剔除)")
    
    # 冗余对
    if not df_redundant.empty:
        print(f"⚠️ 高度冗余的因子对（|corr| > 0.7）: {len(df_redundant)} 组")
        print("   (建议保留其中 IC/ICIR 更高的因子)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())