# -*- coding: utf-8 -*-
"""
因子评估与筛选模块（Factor Evaluation，训练集 ICIR 粗筛）

功能：
    1. 读取预处理产物 features.parquet + labels.parquet（已去极值+行业/市值
       中性化+标准化），按 symbol+date 合并
    2. 【铁律】仅使用训练集（TRAIN_PERIOD 2016-01-01 ~ 2023-04-30）计算
       日频 RankIC，绝不含验证/测试期数据（验证集仅用于模型选择，
       测试集只在最终评估用一次）
    3. 汇总统计：IC 均值、IC 标准差、ICIR、t 值、p 值
    4. 输出 Top 20 / Bottom 10 因子
    5. 保存：factor_evaluation.parquet（IC 汇总）+ daily_rankic.parquet（日频 IC）

输入：data/processed/features.parquet（中性化后特征）+ labels.parquet（原值标签）
输出：data/processed/factor_evaluation.parquet / daily_rankic.parquet

目标参考值（A股实践）：
    - IC 均值 ≥ 0.03 即为可用（0.10 是极优秀水平，不必强求）
    - ICIR ≥ 0.08 为稳健（IR = IC均值 / IC标准差）
    - p 值 < 0.05 为统计显著（注意多重检验校正）
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

# 常数序列计算 Spearman 时无定义，corrwith 会返回 NaN，此警告无害，抑制刷屏
warnings.filterwarnings("ignore", category=UserWarning, message=r".*input array is constant.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*input array is constant.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROCESSED,
    FEATURES_FILE,
    LABELS_FILE,
    TRAIN_PERIOD,
)

# 非因子列（主键/元数据/标记），训练时不计入 IC
META_COLS = {"symbol", "date", "industry", "limit_up", "limit_down",
             "lock_limit_up", "lock_limit_down", "log_mv"}


def compute_daily_rankic(df: pd.DataFrame, factor_cols: list, label_col: str = "label_ret_5"):
    """
    按交易日分组，计算每个因子的日频 RankIC（Spearman 相关系数）。
    
    返回：DataFrame，索引为日期，列为因子名，值为当日 IC。
    """
    daily_ic_list = []
    
    for date, group in df.groupby("date", sort=True):
        # 剔除该日 label 缺失的样本（标签必须已知，否则无法计算相关系数）
        valid_mask = group[label_col].notna()
        if valid_mask.sum() < 5:  # 至少需要 5 只股票才有统计意义
            continue
            
        sub = group[valid_mask]
        # 计算该日所有因子与 label 的 Spearman 相关系数
        ic_series = sub[factor_cols].corrwith(sub[label_col], method="spearman")
        ic_series.name = date
        daily_ic_list.append(ic_series)
    
    daily_ic_df = pd.DataFrame(daily_ic_list)
    return daily_ic_df


def summarize_ic(
    daily_ic: pd.DataFrame,
    factor_cols: list,
    min_samples: int = 30,  # 至少需要 30 个交易日才纳入统计
) -> pd.DataFrame:
    """
    基于日频 IC 表计算汇总统计（IC 均值/标准差/ICIR/t 值/p 值）。
    供全量评估（main）统一调用；单批评估也可复用。
    """
    # 剔除出现次数过少的因子（大部分因子应该有完整序列）
    valid_factors = daily_ic.columns[daily_ic.count() >= min_samples]
    daily_ic = daily_ic[valid_factors]

    # 计算汇总统计
    summary = pd.DataFrame({
        "ic_mean": daily_ic.mean(),
        "ic_std": daily_ic.std(),
        "ic_count": daily_ic.count(),
    })

    # ICIR（IC 均值 / IC 标准差）
    summary["icir"] = summary["ic_mean"] / (summary["ic_std"] + 1e-12)

    # t 值和 p 值（原假设：IC 均值为 0）
    t_stats, p_vals = ttest_1samp(daily_ic, 0, nan_policy="omit")
    summary["t_stat"] = t_stats
    summary["p_value"] = p_vals

    # 5. 补充因子名称（方便查看）
    summary = summary.reset_index().rename(columns={"index": "factor"})
    
    # 按 IC 绝对值排序（越靠前越可能是真实 Alpha）
    summary = summary.sort_values("ic_mean", ascending=False).reset_index(drop=True)
    
    return summary, daily_ic


def main():
    parser = argparse.ArgumentParser(description="训练集日频 RankIC 全量评估")
    parser.add_argument("--label", default="label_ret_5",
                        choices=("label_ret_5", "label_ret_10", "label_ret_20"))
    parser.add_argument("--output-prefix", default="",
                        help="非空时写入 <prefix>_factor_evaluation/daily_rankic，避免覆盖V1")
    args = parser.parse_args()
    # 1. 加载预处理产物（中性化+标准化后的特征与标签）
    features_path = DATA_PROCESSED / FEATURES_FILE
    labels_path = DATA_PROCESSED / LABELS_FILE
    if not features_path.exists() or not labels_path.exists():
        print(f"[ERROR] 缺少预处理产物 {FEATURES_FILE}/{LABELS_FILE}，请先运行 main.py preprocess_factors")
        return 1

    import pyarrow.parquet as pq

    label_col = args.label
    if label_col not in pq.read_schema(labels_path).names:
        print(f"[ERROR] 标签列 {label_col} 不存在于 {LABELS_FILE}，请检查预处理脚本")
        return 1

    # 特征列：features 表中除元数据外的全部列（预处理后已中性化+标准化）
    schema = pq.read_schema(features_path)
    feature_cols = [c for c in schema.names if c not in META_COLS]
    n_rows = pq.ParquetFile(features_path).metadata.num_rows
    print(f"📊 预处理特征表: {n_rows:,} 行, {len(feature_cols)} 个因子")

    # 2. 加载标签并只保留训练集（铁律：粗筛只用训练集，绝不含验证/测试期）
    labels = pd.read_parquet(labels_path, columns=["symbol", "date", label_col])
    labels["date"] = pd.to_datetime(labels["date"])
    train_start, train_end = TRAIN_PERIOD
    labels = labels[(labels["date"] >= train_start) & (labels["date"] <= train_end)]
    print(f"🔒 训练集范围: {train_start} ~ {train_end}（排除验证/测试期，防泄漏）")
    print(f"   训练集标签: {len(labels):,} 行, {labels['date'].nunique()} 个交易日")

    # 3. 按 row group 分块读取 features（16GB 大表，流式处理控峰值内存），
    #    每块只保留训练集日期 -> 合并标签 -> 累积日频 IC 序列
    #    利用 row group 的 date 列统计信息跳过验证/测试期块（铁律防泄漏 + 省 I/O）
    daily_ic_parts = []
    pf = pq.ParquetFile(features_path)
    n_rg = pf.metadata.num_row_groups
    t0 = time.time()
    for i in range(n_rg):
        rg = pf.metadata.row_group(i)
        date_col = rg.column(1)  # date 列
        if date_col.statistics is not None:
            rg_max = date_col.statistics.max
            rg_min = date_col.statistics.min
            # 整块在训练集之前/之后：跳过（之前不存在；之后即验证/测试期）
            if rg_min > pd.Timestamp(train_end):
                continue
            if rg_max < pd.Timestamp(train_start):
                continue
        chunk = pf.read_row_group(i, columns=["symbol", "date"] + feature_cols).to_pandas()
        chunk["date"] = pd.to_datetime(chunk["date"])
        # 只保留训练集日期（row group 内部也按 date 过滤，边界块会被精确裁剪）
        chunk = chunk[(chunk["date"] >= train_start) & (chunk["date"] <= train_end)]
        if chunk.empty:
            continue
        merged = chunk.merge(labels, on=["symbol", "date"], how="inner")
        # 当日 IC 计算：至少 5 只股票才有统计意义
        daily_ic_batch = compute_daily_rankic(merged, feature_cols, label_col)
        daily_ic_parts.append(daily_ic_batch)
        del chunk, merged
        gc.collect()
        if (i + 1) % 10 == 0 or i == n_rg - 1:
            print(f"   ⏳ 已处理 row group {i + 1}/{n_rg}（{time.time() - t0:.0f}s）")
    del pf
    gc.collect()

    # 4. 合并各分块的日频 IC（按行堆叠：列=因子自动对齐，行=各块交易日并集）
    #    注意：不能 concat(axis=1)——列名相同时会产生重复列（270×块数），
    #    且各块日期不重叠（每块覆盖独立时间片），axis=0 直接拼接即可
    daily_ic = pd.concat(daily_ic_parts, axis=0)
    daily_ic = daily_ic[~daily_ic.index.duplicated(keep="first")].sort_index()
    del daily_ic_parts
    gc.collect()

    # 5. 汇总统计（复用评估逻辑：均值/标准差/ICIR/t/p）
    #    注意：summarize_ic 返回 (summary_df, daily_ic_df) 元组，需解包
    summary, _ = summarize_ic(daily_ic, feature_cols)
    print(f"✅ 有效因子数: {len(summary)}")

    # 6. 打印报告
    print("\n" + "=" * 80)
    print("📋 因子评估报告（训练集，按 IC 均值降序）")
    print("=" * 80)
    
    # 只显示前 20 名和后 10 名
    top20 = summary.head(20)
    bottom10 = summary.tail(10)
    
    print("\n🏆 Top 20 因子（IC 均值最高）:")
    print(top20[["factor", "ic_mean", "ic_std", "icir", "p_value"]].to_string(index=False, float_format="%.4f"))
    
    print("\n📉 Bottom 10 因子（IC 均值最低）:")
    print(bottom10[["factor", "ic_mean", "ic_std", "icir", "p_value"]].to_string(index=False, float_format="%.4f"))
    
    # 7. 统计达标情况（根据你设定的目标）
    ic_threshold = 0.10
    icir_threshold = 0.08
    
    ic_ok = summary[summary["ic_mean"] >= ic_threshold]
    icir_ok = summary[summary["icir"] >= icir_threshold]
    
    print("\n📊 达标统计:")
    print(f"   IC ≥ {ic_threshold} 的因子数: {len(ic_ok)} / {len(summary)}")
    print(f"   ICIR ≥ {icir_threshold} 的因子数: {len(icir_ok)} / {len(summary)}")
    
    # 8. 保存评估结果
    stem = f"{args.output_prefix}_" if args.output_prefix else ""
    out_path = DATA_PROCESSED / f"{stem}factor_evaluation.parquet"
    summary.to_parquet(out_path, index=False)
    print(f"\n✅ 评估结果已保存至: {out_path}")
    
    # 保存日频 IC 序列（用于后续时间序列分析 / 相关性剔除）
    ic_series_path = DATA_PROCESSED / f"{stem}daily_rankic.parquet"
    daily_ic.to_parquet(ic_series_path)
    print(f"✅ 日频 IC 序列已保存至: {ic_series_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
