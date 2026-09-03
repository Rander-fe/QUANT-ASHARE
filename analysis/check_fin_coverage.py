# -*- coding: utf-8 -*-
"""检查 extra 底表各财务字段的真实覆盖（非零率），量化 fillna(0) 的影响"""
import pandas as pd

df = pd.read_parquet("c:/QUANT-ASHARE/data/processed/basic_cleaned_with_extra_by_date.parquet")

print("字段               非零率     非NaN率     有效(非0非NaN)行数")
print("-" * 70)
for col in ["roe", "roa", "netprofit_yoy", "grossprofit_margin",
            "netprofit_margin", "debt_to_assets", "current_ratio", "eps",
            "pe_ttm", "pb", "ps_ttm", "total_mv", "turnover_rate"]:
    s = df[col]
    nz = (s != 0).mean()
    nn = s.notna().mean()
    print(f"{col:22s} {nz:10.4f} {nn:10.4f} {int((s != 0).sum()):>12,}")

# 非零 roe 的行里股票数（真实有财务数据的股票）
nz_symbols = df.loc[df["roe"] != 0, "symbol"].nunique()
print(f"\nroe != 0 的股票数: {nz_symbols} / {df['symbol'].nunique()}")
