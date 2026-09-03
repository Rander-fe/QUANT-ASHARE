# -*- coding: utf-8 -*-
"""临时验证：price_structure 因子对齐/未来函数/范围（用 2024 子集加速）"""
import sys
sys.path.insert(0, "c:/QUANT-ASHARE")

import numpy as np
import pandas as pd
from config.settings import DATA_PROCESSED

from factors.base.price_structure import (
    calc_bias, calc_pos_ma, calc_upper_shadow, calc_lower_shadow,
    calc_gap, calc_ma_align, calc_close_pos,
)

df = pd.read_parquet(DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df = df[df["date"] >= "2024-01-01"].copy()
print(f"数据(2024+): {len(df):,} 行, {df['symbol'].nunique()} 只股票")

# ---- 1. 对齐验证：全局 vs 前 3 只股票单独计算 ----
syms = df["symbol"].unique()[:3]
sub = df[df["symbol"].isin(syms)].copy()

bias = calc_bias(df, 10)
bias_sub = calc_bias(sub, 10)
print(f"\n[BIAS10 对齐] 最大差异: {(bias.loc[sub.index]-bias_sub).abs().max():.2e}  一致: {np.allclose(bias.loc[sub.index], bias_sub, equal_nan=True)}")

gap = calc_gap(df, 5)
gap_sub = calc_gap(sub, 5)
print(f"[GAP5 对齐] 最大差异: {(gap.loc[sub.index]-gap_sub).abs().max():.2e}  一致: {np.allclose(gap.loc[sub.index], gap_sub, equal_nan=True)}")

up = calc_upper_shadow(df, 10)
up_sub = calc_upper_shadow(sub, 10)
print(f"[UPPER_SHADOW10 对齐] 最大差异: {(up.loc[sub.index]-up_sub).abs().max():.2e}  一致: {np.allclose(up.loc[sub.index], up_sub, equal_nan=True)}")

# ---- 2. 手工验证单股 ----
sym0 = syms[0]
g = df[df["symbol"] == sym0].reset_index(drop=True)
ma10 = g["close"].rolling(10, min_periods=10).mean()
bias10_manual = (g["close"] - ma10) / (ma10.abs() + 1e-12)
bias10_calc = bias.loc[df["symbol"] == sym0].reset_index(drop=True)
print(f"\n[BIAS10 手工] 差异: {(bias10_manual-bias10_calc).abs().max():.2e}")

prev_close = g["close"].shift(1)
gap5_manual = ((g["open"] - prev_close) / (prev_close + 1e-12)).rolling(5, min_periods=1).mean()
gap5_calc = gap.loc[df["symbol"] == sym0].reset_index(drop=True)
print(f"[GAP5 手工] 差异: {(gap5_manual-gap5_calc).abs().max():.2e}")

# ---- 3. 未来函数：截断到 T，比较 T 日因子值 ----
T = pd.Timestamp("2024-06-28")
df_trunc = df[df["date"] <= T].copy()
bias_T_full = calc_bias(df, 10).loc[df["date"] == T]
bias_T_trunc = calc_bias(df_trunc, 10).loc[df_trunc["date"] == T]
print(f"\n[未来函数 BIAS10] T={T.date()} 差异: {(bias_T_full-bias_T_trunc).abs().max():.2e}")

gap_T_full = calc_gap(df, 5).loc[df["date"] == T]
gap_T_trunc = calc_gap(df_trunc, 5).loc[df_trunc["date"] == T]
print(f"[未来函数 GAP5] 差异: {(gap_T_full-gap_T_trunc).abs().max():.2e}")

# ---- 4. 范围检查 ----
pos = calc_pos_ma(df, 20)
cp = calc_close_pos(df, 20)
us = calc_upper_shadow(df, 10)
ls = calc_lower_shadow(df, 10)
print(f"\n[POS_MA20] 范围 [{pos.min():.4f}, {pos.max():.4f}]")
print(f"[CLOSE_POS20] 范围 [{cp.min():.4f}, {cp.max():.4f}]")
print(f"[UPPER_SHADOW10] 范围 [{us.min():.4f}, {us.max():.4f}]")
print(f"[LOWER_SHADOW10] 范围 [{ls.min():.4f}, {ls.max():.4f}]")
print(f"[GAP5] 范围 [{calc_gap(df,5).min():.4f}, {calc_gap(df,5).max():.4f}]")
print(f"[MA_ALIGN] 范围 [{calc_ma_align(df).min():.4f}, {calc_ma_align(df).max():.4f}]")

# ---- 5. 缺失率 ----
for name, s in [("BIAS10", bias), ("POS_MA20", pos), ("UPPER_SHADOW10", us), ("GAP5", gap)]:
    print(f"\n[{name}] 缺失率 {s.isna().mean()*100:.2f}%  非空 {s.notna().sum():,}")
