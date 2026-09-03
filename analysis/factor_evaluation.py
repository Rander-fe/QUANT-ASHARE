# -*- coding: utf-8 -*-
"""
因子有效性分析（环节四）

功能：
    1. 计算每日 RankIC（Spearman 相关系数）
    2. 汇总 IC 均值、ICIR、t 值、p 值
    3. 10 分组单调性检验（分组收益 + 多空收益）
    4. 因子相关性矩阵，标记冗余对（|corr| > 0.7）
    5. 输出筛选建议：保留因子清单

输入：data/processed/factors.parquet（含所有因子 + 标签）
输出：
    - data/processed/factor_evaluation.parquet（IC/IR 汇总表）
    - data/processed/factor_monotonicity.parquet（分组收益表）
    - data/processed/factor_redundancy.parquet（冗余对清单）
    - data/processed/recommended_factors.txt（推荐保留因子列表）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED


def compute_daily_rankic(df: pd.DataFrame, factor_cols: list, label_col: str = "label_ret_5"):
    """按日计算每个因子的 RankIC"""
    daily_ic_list = []
    for date, group in df.groupby("date", sort=True):
        valid = group[label_col].notna()
        if valid.sum() < 10:
            continue
        sub = group[valid]
        ic_series = sub[factor_cols].corrwith(sub[label_col], method="spearman")
        ic_series.name = date
        daily_ic_list.append(ic_series)
    return pd.DataFrame(daily_ic_list)


def evaluate_factors(
    df: pd.DataFrame,
    factor_cols: list,
    label_col: str = "label_ret_5",
    min_days: int = 30,
):
    """主评估函数"""
    # 1. 日频 RankIC
    print("📊 计算日频 RankIC...")
    daily_ic = compute_daily_rankic(df, factor_cols, label_col)
    # 剔除出现次数不足 min_days 的因子
    valid = daily_ic.count() >= min_days
    daily_ic = daily_ic.loc[:, valid]
    factor_cols = daily_ic.columns.tolist()
    print(f"   有效因子数: {len(factor_cols)}")

    # 2. 汇总统计
    summary = pd.DataFrame({
        "ic_mean": daily_ic.mean(),
        "ic_std": daily_ic.std(),
        "ic_count": daily_ic.count(),
    })
    summary["icir"] = summary["ic_mean"] / (summary["ic_std"] + 1e-12)

    # t 检验
    t_stats, p_vals = ttest_1samp(daily_ic, 0, nan_policy="omit")
    summary["t_stat"] = t_stats
    summary["p_value"] = p_vals

    # FDR 校正（Benjamini-Hochberg）
    _, p_corrected, _, _ = multipletests(summary["p_value"], alpha=0.05, method="fdr_bh")
    summary["p_value_adj"] = p_corrected

    summary = summary.reset_index().rename(columns={"index": "factor"})
    summary = summary.sort_values("ic_mean", ascending=False).reset_index(drop=True)

    # 3. 分组单调性（对前 N 个因子计算，节省时间，这里全量计算）
    print("📊 计算 10 分组单调性...")
    mono_rows = []
    for factor in factor_cols:
        # 按日分组，取 10 等分
        daily_groups = df.groupby("date")
        group_means = {f"D{i+1}": [] for i in range(10)}
        spread_list = []
        for _, group in daily_groups:
            if group[factor].isna().sum() > 0.5 * len(group):
                continue
            group["rank_pct"] = group[factor].rank(pct=True)
            group["decile"] = pd.cut(group["rank_pct"], bins=10, labels=[f"D{i+1}" for i in range(10)])
            means = group.groupby("decile")[label_col].mean()
            for g in [f"D{i+1}" for i in range(10)]:
                if g in means.index:
                    group_means[g].append(means[g])
                else:
                    group_means[g].append(np.nan)
            if "D1" in means.index and "D10" in means.index:
                spread_list.append(means["D1"] - means["D10"])
        # 各组的平均收益
        d_means = {g: np.nanmean(group_means[g]) for g in [f"D{i+1}" for i in range(10)]}
        mono_rows.append({
            "factor": factor,
            "long_short": np.nanmean(spread_list),
            "D1": d_means["D1"],
            "D2": d_means["D2"],
            "D3": d_means["D3"],
            "D4": d_means["D4"],
            "D5": d_means["D5"],
            "D6": d_means["D6"],
            "D7": d_means["D7"],
            "D8": d_means["D8"],
            "D9": d_means["D9"],
            "D10": d_means["D10"],
        })
    df_mono = pd.DataFrame(mono_rows)

    # 4. 相关性矩阵（仅对通过 IC 和单调性初步筛选的因子）
    print("📊 计算因子相关性...")
    # 选 IC_mean > 0.01 且 long_short > 0 的因子做相关性分析，避免噪音过多
    good_factors = summary[summary["ic_mean"] > 0.01].merge(
        df_mono[df_mono["long_short"] > 0], on="factor"
    )["factor"].tolist()
    if len(good_factors) > 1:
        corr_mat = df[good_factors].corr(method="spearman")
        redundant_pairs = []
        for i in range(len(good_factors)):
            for j in range(i+1, len(good_factors)):
                f1, f2 = good_factors[i], good_factors[j]
                corr = corr_mat.loc[f1, f2]
                if abs(corr) > 0.7:
                    redundant_pairs.append({"factor_1": f1, "factor_2": f2, "correlation": corr})
        df_red = pd.DataFrame(redundant_pairs)
    else:
        df_red = pd.DataFrame(columns=["factor_1", "factor_2", "correlation"])

    # 5. 推荐保留清单
    # 规则：IC_mean > 0.02, ICIR > 0.05, p_value_adj < 0.05, long_short > 0, 且不在冗余对的低IC端
    recommended = summary[
        (summary["ic_mean"] > 0.02) &
        (summary["icir"] > 0.05) &
        (summary["p_value_adj"] < 0.05)
    ].merge(df_mono[df_mono["long_short"] > 0], on="factor")
    recommended = recommended["factor"].tolist()

    # 从冗余对中剔除低IC的（保留IC更高的）
    if not df_red.empty:
        high_ic = summary.set_index("factor")["ic_mean"].to_dict()
        to_drop = set()
        for _, row in df_red.iterrows():
            f1, f2 = row["factor_1"], row["factor_2"]
            if f1 in recommended and f2 in recommended:
                if high_ic.get(f1, 0) < high_ic.get(f2, 0):
                    to_drop.add(f1)
                else:
                    to_drop.add(f2)
        recommended = [f for f in recommended if f not in to_drop]

    return summary, df_mono, df_red, recommended


def main():
    input_path = DATA_PROCESSED / "factors.parquet"
    if not input_path.exists():
        print(f"[ERROR] 未找到 {input_path}")
        return 1

    df = pd.read_parquet(input_path)
    print(f"📊 加载数据: {len(df):,} 行, {len(df.columns)} 列")

    # 确定因子列（排除主键、日期、标签）
    exclude = {"symbol", "date", "label_ret_5", "label_ret_10", "label_ret_20"}
    factor_cols = [c for c in df.columns if c not in exclude]

    # 评估
    summary, df_mono, df_red, recommended = evaluate_factors(df, factor_cols, label_col="label_ret_5")

    # 保存结果
    summary.to_parquet(DATA_PROCESSED / "factor_evaluation.parquet", index=False)
    df_mono.to_parquet(DATA_PROCESSED / "factor_monotonicity.parquet", index=False)
    if not df_red.empty:
        df_red.to_parquet(DATA_PROCESSED / "factor_redundancy.parquet", index=False)

    # 保存推荐清单
    with open(DATA_PROCESSED / "recommended_factors.txt", "w") as f:
        f.write("\n".join(recommended))

    print("\n" + "=" * 80)
    print("📋 因子评估报告摘要")
    print("=" * 80)
    print(f"总因子数: {len(factor_cols)}")
    print(f"有效因子数（有足够交易日）: {len(summary)}")
    print(f"推荐保留因子数: {len(recommended)}")
    if not df_red.empty:
        print(f"冗余因子对: {len(df_red)} 组")
        print(df_red.head())

    print("\n🏆 Top 10 因子（按 IC 均值）:")
    print(summary.head(10)[["factor", "ic_mean", "icir", "p_value_adj"]].to_string(index=False, float_format="%.4f"))

    print(f"\n✅ 推荐因子清单已保存至: {DATA_PROCESSED / 'recommended_factors.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())