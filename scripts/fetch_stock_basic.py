# -*- coding: utf-8 -*-
"""
拉取股票静态信息（上市日期 + 历史名称变更），供 clean_data 做 ST/次新股过滤。

内容：
    1. list_date 上市日期
       主源：tushare namechange 的最早 start_date（真实上市日，2000 年前上市也准确）
       兜底：Qlib instruments/all.txt 的 start 列（数据起点，仅当 namechange 无记录时）
    2. namechange 历史名称变更（Point-in-Time 区间表）
       tushare namechange 接口：每条记录表示 [start_date, end_date) 区间内的股票名称。
       用于 ST/退市股过滤 —— 按 (symbol, date) 取当日生效名称判断，
       避免用"当前名称"误判历史期间的 ST 状态（防未来函数）。

产物：
    data/processed/stock_basic.parquet   静态表：symbol, list_date, name(当前名), is_st_now
    data/processed/namechange.parquet    区间表：symbol, name, start_date, end_date

用法：
    python scripts/fetch_stock_basic.py             # 全量拉取（断点续传）
    python scripts/fetch_stock_basic.py --limit 50  # 只拉 50 只（快速验证）

依赖：tushare token（环境变量 TUSHARE_TOKEN 或根目录 .env）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DAILY_RAW_DIR, DATA_PROCESSED, QLIB_PROVIDER_URI, get_tushare_token

# ========================
# 1. 路径配置
# ========================
STOCK_BASIC_PATH = DATA_PROCESSED / "stock_basic.parquet"
NAMECHANGE_PATH = DATA_PROCESSED / "namechange.parquet"

# 指数代码前缀（上证指数 SH000xxx、深证指数 SZ399xxx），股票池构建时剔除
INDEX_PREFIXES = ("SH000", "SZ399")


def get_stock_list_from_daily(limit: int | None = None) -> list[str]:
    """从已导出的日线数据中读取股票代码列表（Qlib 格式：SH600519），过滤指数"""
    frames = []
    for prefix in ["sh", "sz"]:
        fpath = DAILY_RAW_DIR / f"daily_{prefix}.parquet"
        if fpath.exists():
            frames.append(pd.read_parquet(fpath)[["symbol"]])

    if not frames:
        raise FileNotFoundError("未找到日线数据，请先运行 fetch_daily.py 导出日线数据！")

    df_all = pd.concat(frames, ignore_index=True)
    stock_list = [
        s for s in df_all["symbol"].unique().tolist()
        if not s.startswith(INDEX_PREFIXES)
    ]
    n_index = df_all["symbol"].nunique() - len(stock_list)
    if n_index:
        print(f"   - 已过滤 {n_index} 个指数代码")

    if limit is not None:
        stock_list = stock_list[:limit]

    print(f"📊 股票池：{len(stock_list)} 只")
    return stock_list


def qlib_code_to_ts_code(symbol: str) -> str:
    """Qlib格式 (SH600519) -> Tushare格式 (600519.SH)"""
    if symbol.startswith(("SH", "SZ", "BJ")):
        return f"{symbol[2:]}.{symbol[:2]}"
    return symbol


def load_qlib_instruments() -> pd.DataFrame:
    """读取 Qlib instruments/all.txt（symbol, start, end），start 作为上市日期兜底源"""
    ins_path = Path(QLIB_PROVIDER_URI.replace("~", str(Path.home()))) / "instruments" / "all.txt"
    if not ins_path.exists():
        print(f"   ⚠️ Qlib instruments 不存在: {ins_path}，上市日期兜底源不可用")
        return pd.DataFrame(columns=["symbol", "start"])
    df = pd.read_csv(ins_path, sep="\t", header=None, names=["symbol", "start", "end"])
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    print(f"   - Qlib instruments: {len(df)} 只（start 列作兜底）")
    return df[["symbol", "start"]]


def _call_with_retry(pro, method: str, max_retries: int = 3, **kwargs) -> pd.DataFrame:
    """带重试和指数退避的 Tushare 调用"""
    for attempt in range(max_retries):
        try:
            df = getattr(pro, method)(**kwargs)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries}] {method} -> {e}，{wait}s 后重试")
            time.sleep(wait)
    print(f"    [FAILED] {method}")
    return pd.DataFrame()


def fetch_namechange(pro, stock_list: list[str], sleep: float = 0.35, checkpoint_every: int = 200) -> pd.DataFrame:
    """逐只拉取 namechange，增量落盘实现断点续传（以已有 ts_code 判断已完成）"""
    existing = set()
    if NAMECHANGE_PATH.exists():
        try:
            old = pd.read_parquet(NAMECHANGE_PATH, columns=["symbol"])
            existing.update(old["symbol"].unique().tolist())
        except Exception:
            pass
    todo = [c for c in stock_list if c not in existing]
    print(f"\n💾 断点续传：已有 {len(existing)} 只，本次需拉取 {len(todo)} 只")

    frames: list[pd.DataFrame] = []
    total = len(todo)
    for i, code in enumerate(todo, 1):
        df = _call_with_retry(pro, "namechange", ts_code=qlib_code_to_ts_code(code))
        if not df.empty:
            df["symbol"] = code
            frames.append(df)
        if i % checkpoint_every == 0 or i == total:
            if frames:
                batch = pd.concat(frames, ignore_index=True)
                _merge_and_save(batch)
                frames = []
            print(f"💾 checkpoint 已保存: {i}/{total} 只，中断后可续传")
        time.sleep(sleep)

    # 汇总（含历史已有记录）
    all_frames = [pd.read_parquet(NAMECHANGE_PATH)] if NAMECHANGE_PATH.exists() else []
    if frames:
        all_frames.append(pd.concat(frames, ignore_index=True))
    result = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    # 统一日期类型后再去重
    for col in ["start_date", "end_date"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], errors="coerce")
    result = result.drop_duplicates(subset=["symbol", "start_date"], keep="last")
    return result


def _merge_and_save(batch: pd.DataFrame):
    """把新拉的批次合并进 namechange.parquet（增量去重）"""
    # 统一日期列为 datetime，避免 tushare 字符串与新批次混拼导致类型冲突
    for col in ["start_date", "end_date"]:
        if col in batch.columns:
            batch[col] = pd.to_datetime(batch[col], errors="coerce")
    if NAMECHANGE_PATH.exists():
        old = pd.read_parquet(NAMECHANGE_PATH)
        for col in ["start_date", "end_date"]:
            if col in old.columns:
                old[col] = pd.to_datetime(old[col], errors="coerce")
        merged = pd.concat([old, batch], ignore_index=True)
    else:
        merged = batch
    merged = merged.drop_duplicates(subset=["symbol", "start_date"], keep="last")
    merged.to_parquet(NAMECHANGE_PATH, index=False)


def build_stock_basic(namechange: pd.DataFrame, qlib_ins: pd.DataFrame, stock_list: list[str]) -> pd.DataFrame:
    """构建静态证券信息表：symbol, list_date, name(当前名), is_st_now

    以股票池为基准（只含日线里的股票），list_date 主源为 namechange 最早 start_date
    （真实上市日，2000 年前上市也准确），Qlib instruments start 兜底。
    """
    # 上市日期主源：namechange 最早 start_date
    nc = namechange.copy()
    nc["start_date"] = pd.to_datetime(nc["start_date"], errors="coerce")
    list_date_main = nc.groupby("symbol")["start_date"].min().rename("list_date_main")

    basic = pd.DataFrame({"symbol": stock_list})
    basic = basic.set_index("symbol").join(list_date_main, how="left")
    # Qlib start 兜底（只对在 instruments 中的股票生效）
    qlib_map = qlib_ins.set_index("symbol")["start"].rename("list_date_fallback")
    basic = basic.join(qlib_map, how="left")
    basic["list_date"] = basic["list_date_main"].fillna(basic["list_date_fallback"])
    basic = basic.drop(columns=["list_date_main", "list_date_fallback"])

    # 当前名称与当前 ST 状态（诊断用）
    cur = namechange.sort_values("start_date").drop_duplicates("symbol", keep="last")
    cur = cur.rename(columns={"name": "name"})[["symbol", "name"]]
    basic = basic.reset_index().merge(cur, on="symbol", how="left")
    basic["is_st_now"] = basic["name"].str.contains("ST|退", na=False, case=False)

    missing = int(basic["list_date"].isna().sum())
    if missing:
        print(f"   ⚠️ {missing} 只股票无 list_date（无 namechange 且不在 Qlib instruments）")
    basic = basic.dropna(subset=["list_date"])
    print(f"📋 静态信息表：{len(basic)} 只（以股票池为基准）")
    print(f"   当前名称含 ST/退 的股票：{int(basic['is_st_now'].sum())} 只")
    return basic


def main(limit: int | None = None) -> int:
    print("=" * 60)
    print("📋 开始拉取股票静态信息（上市日期 + 历史名称变更）...")
    print("=" * 60)

    token = get_tushare_token()
    if not token:
        print("[ERROR] 未找到 TUSHARE_TOKEN（环境变量或 .env）")
        return 1
    pro = ts.pro_api(token)

    # 1. 股票池
    stock_list = get_stock_list_from_daily(limit=limit)

    # 2. Qlib instruments（list_date 兜底源）
    qlib_ins = load_qlib_instruments()

    # 3. 拉取 namechange（断点续传）
    namechange = fetch_namechange(pro, stock_list)

    # 4. 构建静态信息表并落盘
    basic = build_stock_basic(namechange, qlib_ins, stock_list)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    basic.to_parquet(STOCK_BASIC_PATH, index=False)

    # namechange 区间表落盘（统一列序与日期类型）
    nc_out = namechange.copy()
    for col in ["start_date", "end_date"]:
        if col in nc_out.columns:
            nc_out[col] = pd.to_datetime(nc_out[col], errors="coerce")
    keep = [c for c in ["symbol", "name", "start_date", "end_date"] if c in nc_out.columns]
    nc_out = nc_out[keep].drop_duplicates(subset=["symbol", "start_date"], keep="last")
    nc_out.to_parquet(NAMECHANGE_PATH, index=False)

    print(f"\n✅ 证券信息拉取完成！")
    print(f"   {STOCK_BASIC_PATH}（{len(basic):,} 只）")
    print(f"   {NAMECHANGE_PATH}（{len(nc_out):,} 条名称区间）")
    print(f"   日期范围: {basic['list_date'].min().date()} ~ {basic['list_date'].max().date()}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="拉取股票静态信息（上市日期 + 历史名称）")
    parser.add_argument("--limit", type=int, default=None, help="只拉取前 N 只（测试用）")
    args = parser.parse_args()
    sys.exit(main(limit=args.limit))
