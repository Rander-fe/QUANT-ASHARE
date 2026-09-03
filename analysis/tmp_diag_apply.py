# -*- coding: utf-8 -*-
"""诊断 _group_apply 返回索引结构"""
import sys
sys.path.insert(0, "c:/QUANT-ASHARE")
import numpy as np
import pandas as pd

# 构造模拟数据：3 只股票，各 5 天，已按 symbol,date 排序
rows = []
for s in ["600000", "000001", "300750"]:
    for d in range(5):
        rows.append({"symbol": s, "date": d, "open": 1.0+d, "high": 2.0+d, "low": 0.5+d, "close": 1.5+d})
df = pd.DataFrame(rows).reset_index(drop=True)
print("原始 df 索引:", df.index.tolist())

def _group_apply(df, func):
    return (
        df.groupby("symbol", group_keys=False)
        .apply(func, include_groups=False)
        .reset_index(level=0, drop=True)
    )

def _calc(group):
    body_top = group[["open", "close"]].max(axis=1)
    rng = group["high"] - group["low"]
    shadow = (group["high"] - body_top) / (rng + 1e-12)
    return shadow.rolling(3, min_periods=1).mean()

res = _group_apply(df, _calc)
print("结果类型:", type(res))
print("结果索引:", res.index.tolist())
print("结果索引 dtype:", res.index.dtype)
print("结果 name:", res.name)
print()
print(res)
print()

# 直接看 groupby.apply 中间结果（不加 reset_index）
mid = df.groupby("symbol", group_keys=False).apply(_calc, include_groups=False)
print("中间结果索引:", mid.index.tolist())
print("中间结果 index names:", mid.index.names)
print(mid)
