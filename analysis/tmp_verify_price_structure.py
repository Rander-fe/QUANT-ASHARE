# -*- coding: utf-8 -*-
"""临时验证脚本：检查 price_structure 因子的正确性（对齐/未来函数/取值范围）"""
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
print(f"数据: {len(df):,} 行, {df['symbol'].nunique()} 只股票, {df['date'].min()} ~ {df['date'].max()}")

# 选 3 只股票做对齐验证
syms = df["symbol"].unique()[:3]
sub = df[df["symbol"].isin(syms)].copy()

# ---- 1. 对齐验证：calc_bias（走 _group_transform 路径）----
bias = calc_bias(df, 10)
bias_sub = calc_bias(sub, 10)
# 逐行对比：全局计算的 bias 在 sub 股票上应与单独计算的 bias_sub 一致
bias_global = bias.loc[sub.index]
diff = (bias_global - bias_sub).abs()
print(f"\n[BIAS 对齐] 全局 vs 单独计算 最大差异: {diff.max():.2e}  是否完全一致: {np.allclose(bias_global, bias_sub, equal_nan=True)}")

# ---- 2. 对齐验证：calc_gap（走 _group_apply 路径）----
gap = calc_gap(df, 5)
gap_sub = calc_gap(sub, 5)
gap_global = gap.loc[sub.index]
print(f"[GAP 对齐] 全局 vs 单独计算 最大差异: {(gap_global-gap_sub).abs().max():.2e}  一致: {np.allclose(gap_global, gap_sub, equal_nan=True)}")

# ---- 3. 手工验证 BIAS10（单股）----
sym0 = syms[0]
g = df[df["symbol"] == sym0].reset_index(drop=True)
ma10_manual = g["close"].rolling(10, min_periods=10).mean()
bias10_manual = (g["close"] - ma10_manual) / (ma10_manual.abs() + 1e-12)
bias10_calc = calc_bias(df, 10).loc[df["symbol"] == sym0].reset_index(drop=True)
print(f"\n[BIAS10 手工验证] 与手算最大差异: {(bias10_manual - bias10_calc).abs().max():.2e}")

# ---- 4. 手工验证 GAP5（单股）----
prev_close = g["close"].shift(1)
gap_manual_raw = (g["open"] - prev_close) / (prev_close + 1e-12)
gap5_manual = gap_manual_raw.rolling(5, min_periods=1).mean()
gap5_calc = calc_gap(df, 5).loc[df["symbol"] == sym0].reset_index(drop=True)
print(f"[GAP5 手工验证] 与手算最大差异: {(gap5_manual - gap5_calc).abs().max():.2e}")

# ---- 5. 未来函数检查：因子值是否只依赖当日及以前数据 ----
# 构造截断数据（到某日 T），因子在 T 日的值应与完整数据 T 日值一致
T = pd.Timestamp("2024-03-15")
df_trunc = df[df["date"] <= T].copy()
bias_full_T = calc_bias(df, 10).loc[df["date"] == T]
bias_trunc_T = calc_bias(df_trunc, 10).loc[df_trunc["date"] == T]
print(f"\n[未来函数 BIAS10] 完整 vs 截断(<=T) 在 T 日最大差异: {(bias_full_T - bias_trunc_T).abs().max():.2e}")

gap_full_T = calc_gap(df, 5).loc[df["date"] == T]
gap_trunc_T = calc_gap(df_trunc, 5).loc[df_trunc["date"] == T]
print(f"[未来函数 GAP5] 完整 vs 截断(<=T) 在 T 日最大差异: {(gap_full_T - gap_trunc_T).abs().max():.2e}")

# ---- 6. 取值范围检查 ----
pos = calc_pos_ma(df, 20)
close_pos = calc_close_pos(df, 20)
print(f"\n[POS_MA20] 范围: [{pos.min():.4f}, {pos.max():.4f}] (理论 0~1)")
print(f"[CLOSE_POS20] 范围: [{close_pos.min():.4f}, {close_pos.max():.4f}] (理论 0~1)")
print(f"[UPPER_SHADOW10] 范围: [{calc_upper_shadow(df,10).min():.4f}, {calc_upper_shadow(df,10).max():.4f}]")
print(f"[LOWER_SHADOW10] 范围: [{calc_lower_shadow(df,10).min():.4f}, {calc_lower_shadow(df,10).max():.4f}]")
print(f"[GAP5] 范围: [{calc_gap(df,5).min():.4f}, {calc_gap(df,5).max():.4f}]")

# ---- 7. 缺失率统计（前 window-1 天应为 NaN）----
for name, s in [("BIAS10", calc_bias(df,10)), ("POS_MA20", calc_pos_ma(df,20)),
                ("UPPER_SHADOW10", calc_upper_shadow(df,10)), ("GAP5", calc_gap(df,5))]:
    print(f"\n[{name}] 缺失率: {s.isna().mean()*100:.2f}%  非空: {s.notna().sum():,}")
