# -*- coding: utf-8 -*-
"""
最小复现验证：Alpha158 按列分批合并 + 按日期分批流式写盘（build_factors.py 新逻辑）

输入：
    主表   = data/processed/factors.parquet（旧文件 55 列 = 瘦身后主表）
    Alpha  = data/processed/alpha158.parquet（只取前 40 列 + 2024 年数据）

输出：data/processed/test_stream_2024.parquet（验证用，可删除）

验证点：
    1. 预分配 float32 列 + reindex 对齐 + 逐列赋值 的合并逻辑
    2. 单 ParquetWriter 贯穿多个日期批次的流式写盘
    3. 读回后行数/列数/列名与内存中一致
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED

ALPHA_COL_BATCH = 40
DATE_BATCH_SIZE = 60
YEAR = 2024
out_path = DATA_PROCESSED / "test_stream_2024.parquet"

# 1. 主表（旧 factors.parquet 即瘦身后主表）→ 取 2024 年
df = pd.read_parquet(DATA_PROCESSED / "factors.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.year == YEAR]
print(f"📊 主表({YEAR}): {df.shape}")
df = df.set_index(["symbol", "date"]).sort_index()

# 2. Alpha158 前 40 列（含 2024 年数据）
alpha_path = DATA_PROCESSED / "alpha158.parquet"
alpha_schema = pq.read_schema(alpha_path)
all_cols = [c for c in alpha_schema.names if c not in ("symbol", "date")]
alpha_cols = all_cols[:ALPHA_COL_BATCH]
print(f"📊 Alpha158 取前 {len(alpha_cols)} 列: {alpha_cols[:3]} ... {alpha_cols[-3:]}")

df_alpha = pd.read_parquet(alpha_path, columns=["symbol", "date"] + alpha_cols)
df_alpha["date"] = pd.to_datetime(df_alpha["date"])
df_alpha = df_alpha.set_index(["symbol", "date"])
df_alpha = df_alpha[df_alpha.index.get_level_values("date").year == YEAR]
for c in alpha_cols:
    df_alpha[c] = df_alpha[c].astype(np.float32)
print(f"📊 df_alpha({YEAR}): {df_alpha.shape}")

# 3. 合并：预分配 float32 列 + reindex 对齐 + 逐列赋值
for c in alpha_cols:
    df[c] = np.full(len(df), np.nan, dtype=np.float32)
df_alpha = df_alpha.reindex(df.index)
for c in alpha_cols:
    df[c] = df_alpha[c].to_numpy(dtype=np.float32, copy=False)
del df_alpha
gc.collect()
print(f"✅ 合并后: index={df.index.names}, 数据列 {len(df.columns)} 列")

# 4. 写盘：MultiIndex 两级赋值成普通列（仅 2 列，不触发大块分配）→ RangeIndex → 流式写入
df["symbol"] = df.index.get_level_values("symbol")
df["date"] = df.index.get_level_values("date")
df.index = pd.RangeIndex(len(df))

dates = sorted(df["date"].unique())
n_batches = (len(dates) + DATE_BATCH_SIZE - 1) // DATE_BATCH_SIZE
print(f"📊 流式写盘: {len(dates)} 天 → {n_batches} 批")
writer = None
try:
    for i in range(0, len(dates), DATE_BATCH_SIZE):
        date_batch = dates[i : i + DATE_BATCH_SIZE]
        mask = df["date"].isin(date_batch)
        table = pa.Table.from_pandas(df.loc[mask], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
finally:
    if writer is not None:
        writer.close()

# 5. 读回验证
chk = pd.read_parquet(out_path)
print(f"✅ 读回验证: {chk.shape}")
assert "symbol" in chk.columns and "date" in chk.columns, f"缺少 symbol/date 列! {list(chk.columns)[:5]}"
assert len(chk) == len(df), f"行数不一致! {len(chk)} vs {len(df)}"
assert len(chk.columns) == len(df.columns), f"列数不一致! {len(chk.columns)} vs {len(df.columns)}"
assert sorted(chk.columns) == sorted(df.columns), "列名不一致!"
na_ratio = float(chk[alpha_cols].isna().mean().mean())
print(f"   Alpha158 列 NaN 比例: {na_ratio:.4f}（主表无 2024 数据的股票/日期为 NaN，属预期）")
print("🎉 验证通过：MultiIndex 展开写盘 + 流式写盘逻辑正确")
