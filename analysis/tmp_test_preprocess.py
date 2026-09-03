# -*- coding: utf-8 -*-
"""临时冒烟测试：只处理前 N 个 row group，验证预处理逻辑与性能。"""
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_PROCESSED, BASIC_EXTRA_PATH
from preprocessing.preprocess_factors import (
    process_chunk,
    LABEL_COLS,
    KEY_COLS,
)

N_RG = int(sys.argv[1]) if len(sys.argv) > 1 else 2

pf = pq.ParquetFile(str(DATA_PROCESSED / "factors.parquet"))
extra = pd.DataFrame()
meta_cols = []
if BASIC_EXTRA_PATH.exists():
    cols = KEY_COLS + ["industry", "total_mv", "circ_mv",
                       "limit_up", "limit_down", "lock_limit_up", "lock_limit_down"]
    schema_names = pq.ParquetFile(str(BASIC_EXTRA_PATH)).schema.names
    read_cols = [c for c in cols if c in schema_names]
    extra = pq.ParquetFile(str(BASIC_EXTRA_PATH)).read(columns=read_cols).to_pandas()
    extra["date"] = pd.to_datetime(extra["date"])
    if "total_mv" in extra.columns:
        extra["total_mv"] = pd.to_numeric(extra["total_mv"], errors="coerce")
        extra["log_mv"] = (extra["total_mv"] * 1e4).clip(lower=1.0).apply(__import__("numpy").log)
        extra = extra.drop(columns=["total_mv"])
    if "circ_mv" in extra.columns:
        extra = extra.drop(columns=["circ_mv"])
    for c in ["limit_up", "limit_down", "lock_limit_up", "lock_limit_down"]:
        if c not in extra.columns:
            extra[c] = False
    meta_cols = [c for c in ["industry", "log_mv"] + ["limit_up", "limit_down", "lock_limit_up", "lock_limit_down"]
                 if c in extra.columns]
    print(f"extra: {len(extra):,} 行, meta_cols={meta_cols}")

schema_names = pf.schema.names
label_cols = [c for c in LABEL_COLS if c in schema_names]
feature_cols = [c for c in schema_names if c not in KEY_COLS + label_cols]
print(f"特征数: {len(feature_cols)}, 标签: {label_cols}")

t0 = time.time()
for i in range(N_RG):
    chunk = pf.read_row_group(i).to_pandas()
    chunk["date"] = pd.to_datetime(chunk["date"])
    n_before = len(chunk)
    out = process_chunk(chunk, extra, meta_cols, feature_cols, 0.01, 0.99)
    # 抽样检查
    sample = out.sample(min(3, len(out)), random_state=0)
    print(f"RG{i}: {n_before:,} -> {len(out):,} 行, {time.time()-t0:.0f}s")
    print(sample[["symbol", "date", feature_cols[0], feature_cols[50]]].to_string())
    # 检查该组第一天的横截面分布
    d0 = out["date"].min()
    g = out[out["date"] == d0]
    print(f"  首日 {d0.date()}: {len(g)} 只, 特征0 mean={g[feature_cols[0]].mean():.2e} "
          f"std={g[feature_cols[0]].std():.2f}, NaN={g[feature_cols[0]].isna().sum()}")
    if meta_cols:
        print(f"  行业数: {g['industry'].nunique()}, log_mv NaN: {g['log_mv'].isna().sum()}")
print(f"总耗时: {time.time()-t0:.0f}s（{N_RG} 个 row group）")
