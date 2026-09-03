# -*- coding: utf-8 -*-
"""
环节三：数据预处理（特征工程与标准化）

功能：
    1. 加载原始因子宽表（合并了自定义因子与 Alpha158）
    2. 执行每日横截面缩尾（Winsorize，剔除极端值）
    3. 执行每日横截面 Z-Score 标准化（消除量纲）
    4. 剥离标签列（label_ret_5/10/20）
    5. 保存为 features.parquet 和 labels.parquet

输入：data/processed/factors.parquet（或包含因子的宽表）
输出：
    - data/processed/features.parquet  (仅特征，标准化后)
    - data/processed/labels.parquet    (仅标签)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED

# 从 utils 导入横截面工具
from factors.base.utils import winsorize, cs_zscore


def main():
    # 1. 加载原始因子数据
    input_path = DATA_PROCESSED / "factors.parquet"
    if not input_path.exists():
        print(f"[ERROR] 未找到原始因子表: {input_path}")
        return 1

    df = pd.read_parquet(input_path)
    print(f"📊 加载原始因子: {len(df):,} 行, {len(df.columns)} 列")

    # 2. 分离标签与特征
    label_cols = ["label_ret_5", "label_ret_10", "label_ret_20"]
    existing_labels = [c for c in label_cols if c in df.columns]
    
    # 特征列 = 所有非主键、非日期的列，且不在标签列表中
    exclude_cols = {"symbol", "date"} | set(existing_labels)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"   - 特征数: {len(feature_cols)}")
    print(f"   - 标签数: {len(existing_labels)}")

    # 提取标签（原值，暂不处理）
    df_labels = df[["symbol", "date"] + existing_labels].copy()

    # 3. 特征预处理（每日横截面）
    print("\n🧹 执行特征预处理（每日横截面）...")
    df_features = df[["symbol", "date"] + feature_cols].copy()

    # 3.1 缺失值处理：先填充 0（或向前填充，此处按日截面用 0 代表无值）
    df_features[feature_cols] = df_features[feature_cols].fillna(0)

    # 3.2 缩尾处理 (Winsorize) + 标准化 (Z-Score)
    # 使用 groupby("date") 实现每日横截面处理
    def process_daily(group: pd.DataFrame) -> pd.DataFrame:
        # 对每个特征进行处理
        for col in feature_cols:
            if col not in group.columns:
                continue
            # 缩尾：截断 1% 和 99% 分位数
            group[col] = winsorize(group[col], limits=(0.01, 0.99))
            # 标准化：Z-Score
            group[col] = cs_zscore(group[col])
        return group

    # 应用每日处理
    df_features = df_features.groupby("date", group_keys=False).apply(process_daily)

    # 4. 落盘
    # 4.1 特征文件
    feat_out = DATA_PROCESSED / "features.parquet"
    df_features.to_parquet(feat_out, index=False)
    print(f"✅ 特征已保存: {feat_out} ({len(df_features):,} 行, {len(feature_cols)} 个特征)")

    # 4.2 标签文件
    label_out = DATA_PROCESSED / "labels.parquet"
    df_labels.to_parquet(label_out, index=False)
    print(f"✅ 标签已保存: {label_out} ({len(df_labels):,} 行, {len(existing_labels)} 个标签)")

    # 5. 质量检查：打印标准化后的统计信息
    print("\n📋 标准化后特征统计（前 5 个特征）:")
    sample_cols = feature_cols[:5]
    print(df_features[sample_cols].describe().round(4))

    return 0


if __name__ == "__main__":
    sys.exit(main())