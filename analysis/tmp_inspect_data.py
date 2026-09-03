# -*- coding: utf-8 -*-
"""临时脚本：检查数据文件结构"""
import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path(r"c:\QUANT-ASHARE\data\processed")

for name in ["merged_data.parquet", "factors.parquet", "alpha158.parquet"]:
    p = DATA_PROCESSED / name
    if not p.exists():
        print(f"{name}: 不存在")
        continue
    df = pd.read_parquet(p)
    print(f"===== {name} =====")
    print(f"shape: {df.shape}")
    cols = list(df.columns)
    shown = cols[:25]
    print(f"columns({len(cols)}): {shown}{' ...' if len(cols) > 25 else ''}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        print(f"date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"symbol nunique: {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}")
    label_cols = [c for c in df.columns if "label" in c.lower() or "ret_" in c.lower()]
    print(f"label-like cols: {label_cols}")
    print()
