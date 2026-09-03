# -*- coding: utf-8 -*-
"""
因子构建脚本（使用注册表统一管理 + 合并 Alpha158）

流程：
    1. 加载底表（basic_cleaned_with_extra_by_date.parquet 或 basic_cleaned.parquet）
    2. 导入所有因子模块（自动注册）
    3. 遍历注册表，计算所有自定义因子
    4. 生成标签（label_ret_5/10/20）
    5. 加载 Alpha158 因子（如果存在），并合并到主表
    6. 去重：保留自定义因子优先，Alpha158 中重复的列被丢弃
    7. 保存为 factors.parquet，并打印统计

输出：data/processed/factors.parquet（包含所有因子 + 标签）
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_PROCESSED

# ---------------------------------------------------------------------------
# 底表列投影清单（125 列 → 31 列）
# 只加载因子计算实际依赖的列，内存降 ~75%
# 依赖梳理见 factors/base/ 各模块函数体的 df[...] / group[...] 访问
# ---------------------------------------------------------------------------
BASE_COLS = [
    # 主键
    "symbol", "date",
    # 核心行情（technical/volatility/price_structure/reversal_momentum/volume_price/liquidity）
    "open", "high", "low", "close", "volume", "amount",
    # daily_basic
    "turnover_rate", "pe_ttm", "pb", "ps_ttm",
    # 财务-质量（quality.py）
    "roe", "roa", "grossprofit_margin", "netprofit_margin",
    "current_ratio", "debt_to_assets", "roe_waa", "roic", "quick_ratio", "assets_turn",
    # 财务-成长（growth.py：营收同比优先 tr_yoy，兼容 or_yoy）
    "tr_yoy", "or_yoy", "netprofit_yoy", "op_yoy", "ocf_yoy",
    "roe_yoy", "q_sales_yoy", "q_op_qoq",
]

# 读取底表时显式指定的数值列（压缩为 float32，内存再减半）
FLOAT32_COLS = [c for c in BASE_COLS if c not in ("symbol", "date")]

# 导入所有因子模块（自动触发注册）
import factors.base.reversal_momentum  # noqa
import factors.base.volatility  # noqa
import factors.base.liquidity  # noqa
import factors.base.technical  # noqa
import factors.base.value  # noqa
import factors.base.quality  # noqa
import factors.base.growth  # noqa
import factors.base.price_structure  # noqa
import factors.base.volume_price  # noqa
import factors.base.gp_huatai_replication  # noqa

from factors.registry import get_factor_list, get_factor, print_registry_summary


def main():
    # 1. 加载底表（优先使用带 daily_basic 的版本）
    input_path_extra = DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet"
    input_path_base = DATA_PROCESSED / "basic_cleaned.parquet"

    if input_path_extra.exists():
        input_path = input_path_extra
    elif input_path_base.exists():
        input_path = input_path_base
    else:
        print("[ERROR] 未找到底表，请先运行 basic_clean.py 或 fetch_daily_basic.py")
        return 1

    # 列投影：只读因子依赖列（125 → 31），数值列转 float32 减半内存
    df = pd.read_parquet(input_path, columns=BASE_COLS)
    for col in FLOAT32_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    # 排序保证组内时间序正确
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"📊 加载底表: {len(df):,} 行, {len(df.columns)} 列 (列投影后)")
    print(f"   数据范围: {df['date'].min()} ~ {df['date'].max()}")

    # 2. 检查必要字段
    required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] 缺少必要字段: {missing}")
        return 1

    # 3. 按股票分组（因子计算需要）
    grouped = df.groupby("symbol", group_keys=False)

    # 4. 生成标签（多周期）
    df["label_ret_5"] = grouped["close"].transform(lambda x: x.shift(-5) / x - 1)
    df["label_ret_10"] = grouped["close"].transform(lambda x: x.shift(-10) / x - 1)
    df["label_ret_20"] = grouped["close"].transform(lambda x: x.shift(-20) / x - 1)

    # 5. 遍历注册表，计算所有自定义因子
    custom_factor_names = get_factor_list()
    print(f"\n📊 开始计算 {len(custom_factor_names)} 个自定义因子...")

    # 一次性计算所有因子（收集到字典后统一 concat，避免逐列 insert 造成碎片化）
    factor_series = {}
    for name in custom_factor_names:
        info = get_factor(name)
        if info is None:
            continue
        try:
            factor_series[name] = info["func"](df)
        except Exception as e:
            print(f"   ❌ 因子 {name} 计算失败: {e}")

    factor_df = pd.DataFrame(factor_series, index=df.index)
    # 因子列统一 float32（底表已是 float32，合并后整体保持 float32，内存减半）
    factor_df = factor_df.astype(np.float32)
    df = pd.concat([df, factor_df], axis=1)
    print(f"   ✅ 自定义因子计算完成，当前列数: {len(df.columns)}")

    # 5.5 瘦身：因子算完后，底表原始行情/财务列不再需要，只保留必要列以降低 merge 内存
    keep_base = ["symbol", "date", "label_ret_5", "label_ret_10", "label_ret_20"] + custom_factor_names
    df = df[keep_base]
    print(f"   ✅ 瘦身后主表: {len(df.columns)} 列")

    # 5.6 释放底表内存：因子已全部算完，原始行情/财务列已丢弃，
    #     手动触发 GC，为后续 Alpha158 合并腾出内存
    gc.collect()

    # 6. 尝试加载并合并 Alpha158 因子（按列分批，控制峰值内存）
    alpha_path = DATA_PROCESSED / "alpha158.parquet"
    if alpha_path.exists():
        print(f"\n📊 加载 Alpha158 因子: {alpha_path}")
        # 先读 schema 拿列名（不加载数据）
        import pyarrow.parquet as pq

        alpha_schema = pq.read_schema(alpha_path)
        all_alpha_cols = list(alpha_schema.names)
        alpha_factor_cols = [c for c in all_alpha_cols if c not in ["symbol", "date"]]
        print(f"   Alpha158 原始因子数: {len(alpha_factor_cols)}")

        # 去重：剔除与自定义因子同名的列（自定义因子优先）
        overlap = set(alpha_factor_cols) & set(custom_factor_names)
        if overlap:
            print(f"   ⚠️ 发现重复因子 {len(overlap)} 个，将保留自定义版本: {list(overlap)[:5]}...")
            alpha_factor_cols = [c for c in alpha_factor_cols if c not in overlap]

        # 主表转 (symbol, date) MultiIndex 作为对齐基准
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["symbol", "date"])

        # 按列分批读取 Alpha158：任意时刻内存 = 主表 + 一批(~40列)，而非两张全表。
        # 关键：用「逐列赋值」而非 concat。concat 在增量追加时会触发
        # _consolidate_inplace 把全部列合并成单一连续块（此前 195 列 × float32
        # = 7.58 GiB 单块分配失败）；逐列赋值只新建 block，不动已有列，
        # 不触发整表深拷贝，内存增量仅为本批数据。
        batch_size = 40
        merged_count = 0
        for i in range(0, len(alpha_factor_cols), batch_size):
            batch_cols = alpha_factor_cols[i : i + batch_size]
            # 每次只从 parquet 读取 symbol/date + 本批因子列
            df_alpha = pd.read_parquet(alpha_path, columns=["symbol", "date"] + batch_cols)
            df_alpha["date"] = pd.to_datetime(df_alpha["date"])
            df_alpha = df_alpha.set_index(["symbol", "date"])
            # float32 减半内存；reindex 对齐到主表行（how=left 语义，Alpha158 独有行丢弃）
            for col in batch_cols:
                df_alpha[col] = df_alpha[col].astype(np.float32)
            df_alpha = df_alpha.reindex(df.index)
            # 逐列赋值：复用底层 float32 数组，避免整表深拷贝
            for col in batch_cols:
                df[col] = df_alpha[col].to_numpy(dtype=np.float32, copy=False)
            del df_alpha
            gc.collect()
            merged_count += len(batch_cols)
            print(f"   ⏳ 已合并 Alpha158 第 {merged_count}/{len(alpha_factor_cols)} 列...")

        df = df.reset_index()
        print(f"   ✅ 合并后新增 Alpha158 因子: {len(alpha_factor_cols)} 个")
        print(f"   ✅ 合并后总因子数: {len(custom_factor_names) + len(alpha_factor_cols)}")
    else:
        print("⚠️ Alpha158 未生成，跳过合并。如需使用请先运行 build_alpha158.py")
        alpha_factor_cols = []

    # 7. 最终因子列表
    all_factor_names = custom_factor_names + alpha_factor_cols
    print(f"\n📊 最终因子总数: {len(all_factor_names)}")

    # 8. 保存（直接保存 df 本身，避免 df[cols] 列选择触发整表深拷贝）
    #    df 列顺序已是：symbol/date/标签 + 自定义因子 + Alpha158 因子
    out_path = DATA_PROCESSED / "factors.parquet"
    df.to_parquet(out_path, index=False)
    n_rows, n_cols = df.shape

    # 9. 打印统计
    print_registry_summary()
    print(f"\n✅ 合并后因子数据已保存至: {out_path}")
    print(f"   总行数: {n_rows:,}")
    print(f"   总列数: {n_cols}")
    print(f"   因子数: {len(all_factor_names)}")
    print(f"   标签列: label_ret_5, label_ret_10, label_ret_20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
