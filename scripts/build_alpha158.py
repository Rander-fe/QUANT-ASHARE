# -*- coding: utf-8 -*-
"""
用 Qlib 官方 Alpha158 生成 158 个因子（无需手写计算公式）。

Qlib 的 Alpha158 是现成的特征工程处理器，本脚本直接复用其
get_feature_config() 产出的 158 条表达式，通过 qlib.data.D.features
批量计算后落盘为长表 parquet，供下游分析/建模复用。

因子构成（158 个）：
  - 9 个 kbar 因子：KMID/KLEN/KMID2/KUP/KUP2/KLOW/KLOW2/KSFT/KSFT2
  - 4 个价格类因子：OPEN0/HIGH0/LOW0/VWAP0
  - 145 个滚动因子：29 个算子(ROC/MA/STD/BETA/RSQR/RESI/MAX/MIN/QTLU/QTLD/
    RANK/RSV/IMAX/IMIN/IMXD/CORR/CORD/CNTP/CNTN/CNTD/SUMP/SUMN/SUMD/
    VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD) × 5 窗口(5/10/20/30/60)

输出：data/processed/alpha158.parquet（长表：symbol, date, 158 列因子）
运行：C:/Users/haoran/miniconda3/envs/rqalpha/python.exe scripts/build_alpha158.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_PROCESSED, QLIB_PROVIDER_URI

# Qlib 依赖的第三方库版本较老（numpy<2 / gym），静默 gym 横幅避免刷屏
try:
    import gym_notices.notices as _gn

    _gn.notices = {}
except Exception:  # pragma: no cover - 无 gym 时忽略
    pass

import qlib

from qlib.contrib.data.handler import Alpha158
from qlib.data import D

# 每批处理的股票数量（控制内存峰值：约 400 只 × 2500 交易日 × 158 列 float32）
BATCH_SIZE = 400

# 指数 / B股代码识别：上证指数(SH000xxx)、深证指数(SZ399xxx)、B股(SH900/SZ200)
NON_STOCK_PAT = re.compile(r"^(SH000|SZ399|SH900|SZ200)")


def _get_stock_pool(base_symbols: set[str]) -> list[str]:
    """返回底表 ∩ Qlib all 池、且剔除指数/B股后的股票列表（Qlib 格式）。"""
    all_instruments = D.list_instruments(D.instruments("all"), as_list=True)
    pool = [s for s in all_instruments if s in base_symbols and not NON_STOCK_PAT.match(s)]
    return pool


def _features_to_long(feat: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    """把 D.features 的 MultiIndex[(instrument, datetime), cols] 转成长表。"""
    out = feat.reset_index()
    out.columns = ["symbol", "date"] + factor_names
    return out


def main() -> int:
    # 1. 读取底表，确定股票池与日期范围
    base_path = DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet"
    if not base_path.exists():
        print(f"[ERROR] 底表不存在: {base_path}")
        return 1

    df = pd.read_parquet(base_path, columns=["symbol", "date"])
    base_symbols = set(df["symbol"].unique())
    start_date = pd.Timestamp(df["date"].min()).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(df["date"].max()).strftime("%Y-%m-%d")
    print(f"[INFO] 底表范围: {start_date} ~ {end_date}, 共 {len(base_symbols)} 个 symbol")

    # 2. 初始化 Qlib 并获取 Alpha158 表达式
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region="cn")
    handler = Alpha158.__new__(Alpha158)  # 0.9.7 中 get_feature_config 是实例方法
    fields, names = handler.get_feature_config()
    assert len(fields) == len(names) == 158, f"期望 158 个因子, 实际 {len(fields)}"
    print(f"[INFO] Alpha158 表达式共 {len(fields)} 个")

    # 3. 确定股票池
    pool = _get_stock_pool(base_symbols)
    print(f"[INFO] 股票池（底表 ∩ all ∩ 非指数/B股）: {len(pool)} 只")
    removed = sorted(base_symbols - set(pool))
    if removed:
        print(f"[INFO] 剔除指数/B股: {removed}")

    # 4. 分批计算
    frames: list[pd.DataFrame] = []
    n_batches = int(np.ceil(len(pool) / BATCH_SIZE))
    for i in range(0, len(pool), BATCH_SIZE):
        batch = pool[i : i + BATCH_SIZE]
        feat = D.features(batch, fields, start_date, end_date, freq="day")
        long_df = _features_to_long(feat, names)
        frames.append(long_df)
        done = min(i + BATCH_SIZE, len(pool))
        print(f"[INFO] 批次 {i // BATCH_SIZE + 1}/{n_batches} 完成: {done}/{len(pool)} 只, 本批 {len(long_df):,} 行")

    result = pd.concat(frames, ignore_index=True)

    # 5. 类型归一 + 排序 + 落盘
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    out_path = DATA_PROCESSED / "alpha158.parquet"
    result.to_parquet(out_path, index=False)

    print(f"[OK] 已保存 {out_path}")
    print(f"[OK] 形状: {result.shape[0]:,} 行 × {result.shape[1]} 列 (symbol={result['symbol'].nunique()}, 因子={len(names)})")
    print(f"[OK] 日期范围: {result['date'].min().date()} ~ {result['date'].max().date()}")
    print(f"[OK] 缺失率 Top5: {result[names].isna().mean().nlargest(5).round(4).to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
