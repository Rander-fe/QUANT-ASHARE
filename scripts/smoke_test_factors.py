# -*- coding: utf-8 -*-
"""冒烟测试：全市场 2024 年后数据验证所有因子计算"""
import sys
sys.path.insert(0, "c:/QUANT-ASHARE")

import pandas as pd
from config.settings import DATA_PROCESSED

import factors.base.reversal_momentum  # noqa
import factors.base.volatility  # noqa
import factors.base.liquidity  # noqa
import factors.base.technical  # noqa
import factors.base.value  # noqa
import factors.base.quality  # noqa
import factors.base.growth  # noqa

from factors.registry import get_factor_list, get_factor

df = pd.read_parquet(DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
sub = df[df["date"] >= "2024-01-01"].copy()
print(f"测试: {len(sub):,} 行, {sub['symbol'].nunique()} 只股票")

ok, fail = [], []
for name in get_factor_list():
    try:
        s = get_factor(name)["func"](sub)
        n_valid = int(s.notna().sum())
        ok.append((name, n_valid))
    except Exception as e:
        fail.append((name, str(e)[:100]))

print(f"成功 {len(ok)} / 失败 {len(fail)}")
if fail:
    print("--- 失败 ---")
    for n, e in fail:
        print(f"  {n}: {e}")
else:
    print("全部因子计算通过")
    for n, v in ok:
        print(f"  {n:24s} 非空值: {v:>8,}")
