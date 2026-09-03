# -*- coding: utf-8 -*-
"""
通过 tushare pro 拉取 A 股财务三大报表 + 财务指标，并生成 Point-in-Time 事件表。

流程：
    1. 从日线数据中读取股票池
    2. 按股票列表逐只拉取：
       - 利润表 (income)
       - 资产负债表 (balancesheet)
       - 现金流量表 (cashflow)
       - 财务指标 (fina_indicator)
    3. 原始数据落盘到 data/raw/financial/
    4. 生成 Point-in-Time 事件表 data/processed/financial/financial_events.parquet
       （ann_date 对齐、防未来函数；无效日期自动回退 end_date）

用法：
    python scripts/fetch_financial.py               # 拉取 200 只验证 + 生成事件表
    python scripts/fetch_financial.py --limit 50    # 只拉取 50 只
    python scripts/fetch_financial.py --limit all   # 全量拉取
    python scripts/fetch_financial.py --events-only # 不拉数据，仅重建事件表（修复用）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import FIN_RAW_DIR, FIN_PROCESSED_DIR, FIN_REPORT_TYPE, get_tushare_token

# ========================
# 1. 路径配置
# ========================
# fetch_financial.py 顶部，替换硬编码
from config.settings import DAILY_RAW_DIR  # 与 merge_data.py 保持一致

def get_stock_list_from_daily(limit: int | None = None) -> list[str]:
    """从已导出的日线数据中读取股票代码列表（Qlib 格式：SH600519），过滤指数"""
    frames = []
    for prefix in ["sh", "sz"]:
        fpath = DAILY_RAW_DIR / f"daily_{prefix}.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            frames.append(df[["symbol"]])

    if not frames:
        raise FileNotFoundError("未找到日线数据，请先运行 fetch_daily.py 导出日线数据！")

    df_all = pd.concat(frames, ignore_index=True)
    stock_list = df_all["symbol"].unique().tolist()

    # 过滤指数：上证指数 SH000xxx（如沪深300）、深证指数 SZ399xxx（如深证成指）
    stock_list = [
        s for s in stock_list
        if not (s.startswith("SH000") or s.startswith("SZ399"))
    ]
    n_index = len(df_all["symbol"].unique()) - len(stock_list)
    if n_index:
        print(f"   - 已过滤 {n_index} 个指数代码")

    if limit is not None:
        stock_list = stock_list[:limit]

    print(f"📊 从日线数据中加载股票池：{len(stock_list)} 只")
    return stock_list


def qlib_code_to_ts_code(symbol: str) -> str:
    """Qlib格式 (SH600519) -> Tushare格式 (600519.SH)"""
    if symbol.startswith(("SH", "SZ", "BJ")):
        return f"{symbol[2:]}.{symbol[:2]}"
    return symbol


def convert_ts_code_to_symbol(ts_code: str) -> str:
    """Tushare格式 (600519.SH) -> Qlib格式 (SH600519)"""
    try:
        code, market = ts_code.split(".")
        return f"{market.upper()}{code}"
    except ValueError:
        # 如果已经是 Qlib 格式，直接返回
        return ts_code


# ========================
# 2. 带重试的调用函数
# ========================
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


# ========================
# 3. 核心获取函数
# ========================
def _save_checkpoint(result: dict, done: int, total: int):
    """定期把内存中已拉取的数据合并落盘（增量合并去重，可断点续传）"""
    merged = {}
    for key, frames in result.items():
        if frames:
            try:
                merged[key] = pd.concat(frames, ignore_index=True)
            except Exception as e:
                print(f"   ⚠️ checkpoint 合并 {key} 失败: {e}")
                continue
    save_raw_data(merged, FIN_RAW_DIR)
    print(f"💾 checkpoint 已保存: {done}/{total} 只，中断后可续传")


def fetch_financial_data(
    pro,
    stock_list: list[str],
    start_date: str = "20160101",
    end_date: str = "20261231",
    sleep: float = 0.5,
    checkpoint_every: int = 100,
) -> dict[str, pd.DataFrame]:
    """按股票列表逐只拉取财务数据（支持断点续传：跳过已拉取过的股票）"""
    result = {
        "income": [],
        "balancesheet": [],
        "cashflow": [],
        "fina_indicator": [],
    }

    # ---- 断点续传：以 fina_indicator 为准（该表拉取完成 = 该股票已完整拉取）----
    # 注意：不要用其它表（income 等）的并集，否则旧版 fina_indicator 缺 tr_yoy 的
    # 股票会被误判为已拉取，导致 tr_yoy 永远补不上。
    existing = set()
    fin_path = FIN_RAW_DIR / "fina_indicator.parquet"
    if fin_path.exists():
        try:
            old = pd.read_parquet(fin_path, columns=["source_code"])
            existing.update(old["source_code"].unique().tolist())
        except Exception:
            pass
    todo = [c for c in stock_list if c not in existing]
    print(f"\n💾 断点续传：已有 {len(existing)} 只股票，本次需拉取 {len(todo)} 只")

    total = len(todo)
    print(f"\n{'='*60}")
    print(f"📡 开始拉取 {total} 只股票的财务数据...")
    print(f"{'='*60}")

    for i, code in enumerate(todo, 1):
        ts_code = qlib_code_to_ts_code(code)

        try:
            # ---------- 1. 利润表 ----------
            df = _call_with_retry(
                pro, "income",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                report_type=FIN_REPORT_TYPE,
                fields="ts_code,end_date,ann_date,f_ann_date,report_type,comp_type,"
                       "revenue,oper_cost,oper_exp,sell_exp,admin_exp,fin_exp,"
                       "n_income,net_profit,eps,basic_eps,diluted_eps"
            )
            if not df.empty:
                df["source_code"] = code
                result["income"].append(df)

            # ---------- 2. 资产负债表 ----------
            df = _call_with_retry(
                pro, "balancesheet",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                report_type=FIN_REPORT_TYPE,
                fields="ts_code,end_date,ann_date,f_ann_date,report_type,comp_type,"
                       "total_assets,total_liab,debt_to_assets,"
                       "total_equity,retained_earnings,capital_reserve,"
                       "fixed_assets,total_cur_assets,total_cur_liab"
            )
            if not df.empty:
                df["source_code"] = code
                result["balancesheet"].append(df)

            # ---------- 3. 现金流量表 ----------
            df = _call_with_retry(
                pro, "cashflow",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                report_type=FIN_REPORT_TYPE,
                fields="ts_code,end_date,ann_date,f_ann_date,report_type,comp_type,"
                       "c_operate,c_invest,c_finance,"
                       "n_cash_operate,n_cashflow"
            )
            if not df.empty:
                df["source_code"] = code
                result["cashflow"].append(df)

            # ---------- 4. 财务指标 ----------
            # 不指定 fields：返回全部字段（约108列），一次性补齐质量/成长因子所需数据。
            # 注意：revenue_yoy 已被 tushare 下线（即使高积分也拉不到），
            # 用 tr_yoy（营业总收入同比）/ or_yoy（营收同比）替代，两者与 revenue_yoy 同义，
            # 因子层 growth.py 已做兼容读取。
            df = _call_with_retry(
                pro, "fina_indicator",
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            if not df.empty:
                df["source_code"] = code
                result["fina_indicator"].append(df)

            print(f"[{i:4d}/{total}] ✅ {code}")

        except Exception as e:
            print(f"[{i:4d}/{total}] ❌ {code} -> {e}")

        time.sleep(sleep)

        # 定期保存 checkpoint：防止长任务中途中断丢失全部进度
        if checkpoint_every and i % checkpoint_every == 0:
            _save_checkpoint(result, i, total)

    # 合并结果
    merged = {}
    for key, frames in result.items():
        if frames:
            merged[key] = pd.concat(frames, ignore_index=True)
            print(f"   📊 {key}: {len(merged[key]):,} 行")
        else:
            merged[key] = pd.DataFrame()

    return merged


# ========================
# 4. 保存原始数据
# ========================
def save_raw_data(data_dict: dict[str, pd.DataFrame], out_dir: Path):
    """保存原始数据到 Parquet（合并追加：与已有数据按 ts_code+end_date+ann_date 去重）"""
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "income": "income.parquet",
        "balancesheet": "balancesheet.parquet",
        "cashflow": "cashflow.parquet",
        "fina_indicator": "fina_indicator.parquet",
    }

    for key, df in data_dict.items():
        if df.empty:
            print(f"⚠️ {key} 无数据，跳过落盘")
            continue
        fpath = out_dir / file_map.get(key, f"{key}.parquet")

        # 断点续传：若已有旧文件，先读入合并去重再保存，避免覆盖丢失已拉数据
        if fpath.exists():
            try:
                old = pd.read_parquet(fpath)
                dedup_keys = [c for c in ("ts_code", "end_date", "ann_date") if c in old.columns and c in df.columns]
                if dedup_keys:
                    df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                        subset=dedup_keys, keep="last"
                    )
                else:
                    df = pd.concat([old, df], ignore_index=True)
            except Exception as e:
                print(f"   ⚠️ 合并已有 {key} 失败: {e}，直接覆盖")

        df.to_parquet(fpath, index=False)
        print(f"✅ 原始 {key}: {fpath} ({len(df):,} 行)")


# ========================
# 5. 主函数
# ========================
# 5. 生成 Point-in-Time 事件表
# ========================
def build_financial_events() -> int:
    """
    从原始财务指标数据生成 Point-in-Time 事件表。
    防止未来函数：使用 ann_date 对齐，而非 end_date。
    输入：data/raw/financial/fina_indicator.parquet
    输出：data/processed/financial/financial_events.parquet
    """
    fin_path = FIN_RAW_DIR / "fina_indicator.parquet"
    if not fin_path.exists():
        print(f"[ERROR] 未找到 {fin_path}，请先拉取财务数据")
        return 1

    print("=" * 60)
    print("📊 开始生成 Point-in-Time 财务事件表...")
    print(f"   输入: {fin_path}")
    print("=" * 60)

    df = pd.read_parquet(fin_path)
    print(f"📊 加载原始财务指标：{len(df):,} 行，{df.shape[1]} 列")

    # 转换股票代码（优先 ts_code，其次 source_code）
    if "ts_code" in df.columns:
        df["symbol"] = df["ts_code"].apply(convert_ts_code_to_symbol)
    elif "source_code" in df.columns:
        df["symbol"] = df["source_code"]
    else:
        print("[ERROR] 无法找到股票代码列（ts_code 或 source_code）")
        return 1

    # 财报期与公告可用日必须同时存在；绝不用 end_date 代填公告日。
    if "ann_date" not in df.columns or "end_date" not in df.columns:
        print("[ERROR] 缺少 ann_date 或 end_date，无法建立防未来数据的财报期索引")
        return 1
    df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
    df["report_period"] = pd.to_datetime(df["end_date"], errors="coerce")
    invalid = df["ann_date"].isna() | df["report_period"].isna() | (df["report_period"] > df["ann_date"])
    print(f"   - 排除无效/不可能时序: {int(invalid.sum()):,} 行")
    df = df.loc[~invalid].copy()

    # 同日保留较新报告期；旧期迟到修订不能让当前快照倒退。
    df = df.sort_values(["symbol", "ann_date", "report_period"]).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "ann_date"], keep="last")
    print(f"   - 去重: {before - len(df)} 行")
    latest_seen = df.groupby("symbol", sort=False)["report_period"].cummax()
    stale_revision = df["report_period"] < latest_seen
    print(f"   - 排除旧期迟到修订: {int(stale_revision.sum()):,} 行")
    df = df.loc[~stale_revision].copy()
    period_pairs = df[["symbol", "report_period"]].drop_duplicates().sort_values(["symbol", "report_period"])
    period_pairs["report_seq"] = period_pairs.groupby("symbol", sort=False).cumcount().astype("int32")
    df = df.merge(period_pairs, on=["symbol", "report_period"], how="left", validate="many_to_one")

    # 选择核心字段（排除元数据列）
    exclude_cols = [
        "ts_code", "end_date", "source_code", "report_type", "report_period", "report_seq",
        "comp_type", "f_ann_date", "ann_date",
    ]
    keep_cols = ["symbol", "ann_date", "report_period", "report_seq"] + [
        col for col in df.columns
        if col not in exclude_cols and col not in ["symbol", "ann_date", "report_period", "report_seq"]
    ]
    df_events = df[keep_cols].copy()

    print(f"   - 保留字段: {len(keep_cols)} 个")

    # 保存
    FIN_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIN_PROCESSED_DIR / "financial_events.parquet"
    df_events.to_parquet(out_path, index=False)
    index_path = FIN_PROCESSED_DIR / "report_period_index.parquet"
    df_events[["symbol", "ann_date", "report_period", "report_seq"]].to_parquet(index_path, index=False)

    print(f"\n✅ 财务事件表已保存：{out_path}")
    print(f"   共 {len(df_events):,} 行，{df_events['symbol'].nunique()} 只股票")
    print(f"   时间范围：{df_events['ann_date'].min()} ~ {df_events['ann_date'].max()}")
    print(f"   财报期索引：{index_path}")
    return 0


# ========================
# 6. 主函数
# ========================
def main():
    parser = argparse.ArgumentParser(description="拉取 A 股财务数据并生成 Point-in-Time 事件表")
    parser.add_argument(
        "--limit",
        type=str,
        default="200",
        help="拉取股票数量限制：数字 或 'all'（全量拉取）"
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="每次请求间隔秒数")
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="不拉取数据，仅从现有 fina_indicator.parquet 重建事件表（修复用）"
    )
    args = parser.parse_args()

    # 仅重建事件表模式：无需 token
    if args.events_only:
        return build_financial_events()

    import tushare as ts

    # 解析 limit
    if args.limit.lower() == "all":
        limit = None
    else:
        limit = int(args.limit)

    # 获取 token
    token = get_tushare_token()
    if not token:
        print("[ERROR] 未找到 tushare token")
        return 1

    pro = ts.pro_api(token)
    print("✅ Tushare Pro 初始化成功")

    # 获取股票列表
    try:
        stock_list = get_stock_list_from_daily(limit=limit)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    # 拉取数据
    data_dict = fetch_financial_data(
        pro=pro,
        stock_list=stock_list,
        start_date="20160101",
        end_date="20261231",
        sleep=args.sleep,
    )

    # 保存原始数据
    save_raw_data(data_dict, FIN_RAW_DIR)

    # 生成 Point-in-Time 事件表
    return build_financial_events()


if __name__ == "__main__":
    import argparse
    sys.exit(main())
