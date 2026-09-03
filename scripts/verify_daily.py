"""校验日线原始数据完整性：股票覆盖、行数、空值、日期范围。"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_RAW

DAILY_DIR = os.path.join(DATA_RAW, "daily")
FILES = ["daily_bj.parquet", "daily_sh.parquet", "daily_sz.parquet"]


def main() -> None:
    total_stocks = 0
    total_rows = 0
    for f in FILES:
        path = os.path.join(DAILY_DIR, f)
        df = pd.read_parquet(path, columns=["symbol", "date", "close"])
        n = df["symbol"].nunique()
        total_stocks += n
        total_rows += len(df)
        cnt = df.groupby("symbol").size()
        short = (cnt < 100).sum()
        print(f"=== {f} ===")
        print(f"  stocks={n}, rows={len(df)}, min_rows={cnt.min()}, max_rows={cnt.max()}")
        print(f"  <100行股票数={short}, close空值={int(df['close'].isna().sum())}")
        print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
        print()
    print(f"总计: {total_stocks} 只股票, {total_rows:,} 行")


if __name__ == "__main__":
    main()
