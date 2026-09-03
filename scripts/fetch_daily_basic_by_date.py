# -*- coding: utf-8 -*-
"""fetch_daily_basic.py 的副本（实验版）：按交易日批量拉取 daily_basic。

与原版（逐股循环，6000+ 次调用）相比，本版改为「每个交易日一次调用拿全市场」，
大幅降低调用次数（约 2500 次），并加入三处保护以规避 ConnectionResetError：

    1. 失败重试 + 指数退避（应对限流/网络抖动导致的连接重置）
    2. 断点续跑（checkpoint 记录已完成交易日，中断后重跑不从头开始）
    3. 每 N 天落盘一次中间结果，避免内存过大或崩溃丢进度

与原版差异说明：
    - 输出到 data/processed/basic_cleaned_with_extra_by_date.parquet（不覆盖原输出）
    - 中间数据落盘 data/processed/daily_basic_raw.parquet + daily_basic_ckpt.json

注意：daily_basic 按 trade_date 全市场查询需要较高 tushare 积分（通常 2000+）。
    若积分不足，接口会返回空数据或报权限错误，本脚本会在拉取第一个交易日后自检提示。

用法：
    python scripts/fetch_daily_basic_by_date.py                  # 全区间，断点续跑
    python scripts/fetch_daily_basic_by_date.py --limit 10       # 只拉前 10 个交易日（快速验证）
    python scripts/fetch_daily_basic_by_date.py --sleep 0.5      # 自定义间隔秒数
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, get_tushare_token

# daily_basic 需要拉取的字段（按 trade_date 查询时必须包含 ts_code 与 trade_date）
# 注意：daily_basic 的 pct_chg 字段需要较高 tushare 积分（不足时会被静默过滤），
# 官方涨跌幅改用 pro.daily() 接口拉取（该接口按 trade_date 全市场查询积分门槛低）。
DAILY_BASIC_FIELDS = "ts_code,trade_date,total_mv,circ_mv,pe_ttm,pb,ps_ttm,turnover_rate"
# daily 接口的官方涨跌幅（%）——数据端涨跌停识别用，比前复权 close 口径更准
PCT_CHG_FIELDS = "ts_code,trade_date,pct_chg"
# 字段版本标识：任一字段列表变化时，旧 checkpoint 自动失效（强制全量重拉补齐新字段）
FIELDS_VERSION = f"daily_basic:{DAILY_BASIC_FIELDS};daily:{PCT_CHG_FIELDS}"

# 中间产物路径
RAW_PATH = DATA_PROCESSED / "daily_basic_raw.parquet"
CKPT_PATH = DATA_PROCESSED / "daily_basic_ckpt.json"


def ts_code_to_qlib(ts_code: str) -> str:
    """把 tushare 代码转成 qlib 代码：'000001.SZ' -> 'SZ000001'。"""
    code, exch = ts_code.split(".")
    return f"{exch.upper()}{code}"


def load_checkpoint() -> set[str]:
    """读取已完成的交易日集合；字段版本变化时返回空集（强制全量重拉）。"""
    if CKPT_PATH.exists():
        data = json.loads(CKPT_PATH.read_text(encoding="utf-8"))
        if data.get("fields") != FIELDS_VERSION:
            print(
                f"[INFO] 字段版本变化（旧字段集与当前 DAILY_BASIC_FIELDS 不一致），"
                "旧 checkpoint 失效，将全量重拉补齐新字段"
            )
            return set()
        return set(data.get("done", []))
    return set()


def save_checkpoint(done: set[str]) -> None:
    """把已完成的交易日集合 + 字段版本落盘。"""
    CKPT_PATH.write_text(
        json.dumps(
            {"done": sorted(done), "fields": FIELDS_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_raw() -> pd.DataFrame:
    """加载已累积的 daily_basic 长表，无则返回空表。"""
    if RAW_PATH.exists():
        return pd.read_parquet(RAW_PATH)
    return pd.DataFrame()


def save_raw(df: pd.DataFrame) -> None:
    """把累积长表落盘（覆盖写）。"""
    df.to_parquet(RAW_PATH, index=False)


def fetch_one_day(pro, trade_date: str, retries: int, sleep_base: float):
    """拉取单个交易日的全市场 daily_basic + 官方 pct_chg，带重试与指数退避。

    每日两次调用：
      1) pro.daily_basic(trade_date=...) -> 估值字段（total_mv 等）
      2) pro.daily(trade_date=...) -> 官方涨跌幅 pct_chg（%），按 symbol 左合并
    返回 DataFrame（含 symbol/date 及估值字段 + pct_chg）或 None（最终失败）。
    """
    for attempt in range(1, retries + 1):
        try:
            df = pro.daily_basic(
                trade_date=trade_date,
                fields=DAILY_BASIC_FIELDS,
            )
            if df is None or df.empty:
                return pd.DataFrame()  # 空结果（可能非交易日或积分不足）
            df = df.rename(columns={"trade_date": "date"})
            df["symbol"] = df["ts_code"].map(ts_code_to_qlib)
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop(columns=["ts_code"])

            # 官方涨跌幅：daily 接口（daily_basic 的 pct_chg 需更高积分，可能被过滤）
            df_pct = pro.daily(trade_date=trade_date, fields=PCT_CHG_FIELDS)
            if df_pct is not None and not df_pct.empty:
                df_pct = df_pct.rename(columns={"trade_date": "date"})
                df_pct["symbol"] = df_pct["ts_code"].map(ts_code_to_qlib)
                df_pct = df_pct[["symbol", "pct_chg"]]
                df = df.merge(df_pct, on="symbol", how="left")
            else:
                print(f"  ⚠ {trade_date} daily 接口未返回 pct_chg，该日涨跌停标记将缺失")
                df["pct_chg"] = pd.NA
            return df
        except Exception as exc:  # noqa: BLE001 - 网络异常类型繁多，统一兜底重试
            wait = sleep_base * (2 ** (attempt - 1)) * 5  # 指数退避
            print(
                f"  ⚠ {trade_date} 第 {attempt}/{retries} 次失败 "
                f"({type(exc).__name__}: {exc})，{wait:.0f}s 后重试"
            )
            if attempt < retries:
                time.sleep(wait)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="按交易日批量拉取 daily_basic（实验版）")
    parser.add_argument("--limit", type=int, default=0, help="只拉前 N 个交易日（0=全部）")
    parser.add_argument("--sleep", type=float, default=0.3, help="每次调用间隔秒数")
    parser.add_argument("--retries", type=int, default=4, help="单个交易日最大重试次数")
    parser.add_argument("--flush-every", type=int, default=100, help="每 N 天落盘一次中间结果")
    args = parser.parse_args()

    token = get_tushare_token()
    if not token:
        print("[ERROR] 未找到 tushare token")
        return 1
    pro = ts.pro_api(token)

    # 1. 加载 cleaned 底表，推断日期区间
    input_path = DATA_PROCESSED / "basic_cleaned.parquet"
    if not input_path.exists():
        print(f"[ERROR] 未找到 {input_path}，请先运行 clean_data.py")
        return 1
    df = pd.read_parquet(input_path)
    dates = pd.to_datetime(df["date"])
    start, end = dates.min().strftime("%Y%m%d"), dates.max().strftime("%Y%m%d")
    print(f"📊 底表日期区间: {start} ~ {end}，共 {len(df):,} 行")

    # 2. 获取交易日历
    print("\n📅 获取交易日历...")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    # is_open 为 int（1=开市, 0=休市），兼容字符串形式
    trade_dates = sorted(
        cal.loc[cal["is_open"].astype(str).isin(("1", "1.0")), "cal_date"].astype(str).tolist()
    )
    if args.limit > 0:
        trade_dates = trade_dates[: args.limit]
    print(f"   交易日数量: {len(trade_dates)}")

    # 3. 断点续跑：恢复已完成的日期与已累积的数据
    done = load_checkpoint()
    acc = load_raw()
    if not acc.empty and "pct_chg" not in acc.columns:
        print("[INFO] 旧累积数据缺 pct_chg（字段版本升级），丢弃旧数据全量重拉")
        acc = pd.DataFrame()
    if done:
        print(f"♻ 检测到 checkpoint，已跳过 {len(done)} 个交易日")
    if not acc.empty:
        print(f"♻ 已加载累积数据 {len(acc):,} 行")

    # 4. 逐交易日拉取
    print(f"\n📡 开始拉取 daily_basic（间隔 {args.sleep}s）...")
    new_since_flush = 0
    try:
        for i, trade_date in enumerate(trade_dates):
            if trade_date in done:
                continue
            part = fetch_one_day(pro, trade_date, args.retries, args.sleep)
            if part is None:
                print(f"[WARN] {trade_date} 最终失败，已跳过（可重跑补齐）")
                done.add(trade_date)  # 标记完成，避免死循环；如需可手动删 checkpoint 重试
                new_since_flush += 1
            else:
                if not part.empty:
                    acc = pd.concat([acc, part], ignore_index=True)
                done.add(trade_date)
                new_since_flush += 1

            if new_since_flush >= args.flush_every:
                save_raw(acc)
                save_checkpoint(done)
                print(f"  💾 已落盘 checkpoint（累计 {len(done)} 天，{len(acc):,} 行）")
                new_since_flush = 0

            if (i + 1) % 50 == 0:
                print(f"  进度: {i + 1}/{len(trade_dates)}")
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\n⏸ 检测到中断，正在保存进度...")
        save_raw(acc)
        save_checkpoint(done)
        print("进度已保存，重跑本脚本可续跑。")
        return 130

    # 收尾落盘
    save_raw(acc)
    save_checkpoint(done)
    print(f"\n✅ daily_basic 拉取完成: {len(acc):,} 行，覆盖 {len(done)} 个交易日")

    if acc.empty:
        print("[ERROR] 未获取到任何 daily_basic 数据。")
        print("   可能原因：tushare 积分不足，无法按 trade_date 全市场查询 daily_basic。")
        print("   请登录 tushare.pro 确认 daily_basic 接口的调用权限与积分档位。")
        return 1

    # 5. 拉取行业分类（只需一次）
    print("\n📡 拉取行业分类...")
    df_stock = pro.stock_basic(fields="ts_code,symbol,name,industry,list_date")
    df_stock["symbol"] = df_stock["ts_code"].map(ts_code_to_qlib)
    df_industry = df_stock[["symbol", "industry"]].drop_duplicates(subset=["symbol"])
    print(f"✅ 行业分类获取完成: {len(df_industry):,} 只")

    # 6. 合并回主表
    print("\n🔗 合并 daily_basic 和行业分类...")
    df_merged = pd.merge(df, acc, on=["symbol", "date"], how="left")
    df_merged = pd.merge(df_merged, df_industry, on="symbol", how="left")

    fill_cols = ["total_mv", "circ_mv", "pe_ttm", "pb", "ps_ttm", "turnover_rate"]
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)

    # pct_chg 缺失保持 NaN（0 表示平盘，不能用 0 填充，否则会误判涨跌停）
    if "pct_chg" in df_merged.columns:
        df_merged["pct_chg"] = pd.to_numeric(df_merged["pct_chg"], errors="coerce")

    out_path = DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet"
    df_merged.to_parquet(out_path, index=False)
    print(f"✅ 合并完成，保存至: {out_path}")
    print(f"   总行数: {len(df_merged):,}")
    print(f"   新增列: {fill_cols + ['industry', 'pct_chg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
