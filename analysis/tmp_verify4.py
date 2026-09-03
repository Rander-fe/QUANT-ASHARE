# -*- coding: utf-8 -*-
"""验证：未来函数 + POS_MA/CLOSE_POS 重复性"""
import sys
sys.path.insert(0, "c:/QUANT-ASHARE")
import numpy as np
import pandas as pd
from config.settings import DATA_PROCESSED
from factors.base.price_structure import calc_bias, calc_gap, calc_pos_ma, calc_close_pos

df = pd.read_parquet(DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df = df[df["date"] >= "2024-01-01"].reset_index(drop=True).copy()  # 连续索引子集
print(f"子集(2024+): {len(df):,} 行")

# ---- 未来函数：截断到 T，比较 T 日因子值 ----
T = pd.Timestamp("2024-06-28")
df_trunc = df[df["date"] <= T].reset_index(drop=True).copy()

bias_full = calc_bias(df, 10)
bias_trunc = calc_bias(df_trunc, 10)
b_T_full = bias_full.loc[df["date"] == T].reset_index(drop=True)
b_T_trunc = bias_trunc.loc[df_trunc["date"] == T].reset_index(drop=True)
print(f"[未来函数 BIAS10] T={T.date()} 差异 = {(b_T_full-b_T_trunc).abs().max():.2e}")

gap_full = calc_gap(df, 5)
gap_trunc = calc_gap(df_trunc, 5)
g_T_full = gap_full.loc[df["date"] == T].reset_index(drop=True)
g_T_trunc = gap_trunc.loc[df_trunc["date"] == T].reset_index(drop=True)
print(f"[未来函数 GAP5] T={T.date()} 差异 = {(g_T_full-g_T_trunc).abs().max():.2e}")

# ---- POS_MA vs CLOSE_POS 重复性 ----
pos5 = calc_pos_ma(df, 5)
cp5 = calc_close_pos(df, 5)
pos20 = calc_pos_ma(df, 20)
cp20 = calc_close_pos(df, 20)
print(f"\n[重复性] POS_MA5 == CLOSE_POS5: 差异 {(pos5-cp5).abs().max():.2e}, 完全相同={np.allclose(pos5, cp5, equal_nan=True)}")
print(f"[重复性] POS_MA20 == CLOSE_POS20: 差异 {(pos20-cp20).abs().max():.2e}, 完全相同={np.allclose(pos20, cp20, equal_nan=True)}")
