# -*- coding: utf-8 -*-
"""验证：完整 df（build_factors 场景）下 _group_apply 因子是否错位"""
import sys
sys.path.insert(0, "c:/QUANT-ASHARE")
import numpy as np
import pandas as pd
from config.settings import DATA_PROCESSED
from factors.base.price_structure import calc_gap, calc_upper_shadow, calc_lower_shadow

df = pd.read_parquet(DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"完整 df: {len(df):,} 行, 索引连续: {df.index.is_unique and df.index.min()==0 and df.index.max()==len(df)-1}")

# 取 3 只股票，验证手工 vs 因子函数
syms = sorted(df["symbol"].unique())[:3]
print("验证股票:", syms)

# ---- GAP5 手工 vs calc_gap ----
gap = calc_gap(df, 5)
for s in syms:
    mask = df["symbol"] == s
    g = df[mask].reset_index(drop=True)
    prev_close = g["close"].shift(1)
    manual = ((g["open"] - prev_close) / (prev_close + 1e-12)).rolling(5, min_periods=1).mean()
    calc_vals = gap.loc[mask].reset_index(drop=True)
    diff = (manual - calc_vals).abs().max()
    print(f"  GAP5 {s}: 手工 vs 因子 最大差异 = {diff:.2e}  一致={np.allclose(manual, calc_vals, equal_nan=True)}")

# ---- UPPER_SHADOW10 手工 vs calc ----
up = calc_upper_shadow(df, 10)
for s in syms:
    mask = df["symbol"] == s
    g = df[mask].reset_index(drop=True)
    body_top = g[["open", "close"]].max(axis=1)
    rng = g["high"] - g["low"]
    manual = ((g["high"] - body_top) / (rng + 1e-12)).rolling(10, min_periods=1).mean()
    calc_vals = up.loc[mask].reset_index(drop=True)
    diff = (manual - calc_vals).abs().max()
    print(f"  UPPER_SHADOW10 {s}: 手工 vs 因子 最大差异 = {diff:.2e}  一致={np.allclose(manual, calc_vals, equal_nan=True)}")

# ---- 检查 _group_apply 返回索引是否与 df 一致 ----
print(f"\ncalc_gap 返回索引: min={gap.index.min()}, max={gap.index.max()}, 与 df 一致={gap.index.equals(df.index)}")
print(f"calc_upper_shadow 返回索引与 df 一致={up.index.equals(df.index)}")
