# -*- coding: utf-8 -*-
"""
基础数据清洗 —— 将合并后的宽表清洗为“干净但未标准化”的底表。

清洗范围（仅处理数据质量问题，不涉及统计变换）：
    1. 剔除 ST、*ST、退市股（通过名称识别）
    2. 剔除上市不足 60 个交易日的次新股（通过上市日期计算）
    3. 删除重复的 (symbol, date) 记录
    4. 停牌日价格保留 NaN（不填充，对齐 Qlib：停牌 OHLCV=NaN，避免"收益为 0"假象）
    5. 财务缺失值前向填充（FFill）：用距离今日最近的过去数据填充，防止未来函数

注意：
    - 本脚本不进行缩尾（Winsorize）和横截面标准化（Z-Score）。
    - 缩尾和标准化由 Qlib 的 DataHandler 在模型训练时动态处理，
      这样做可以保证“未来信息不泄露”（只拟合训练集）。
    
输入：data/processed/merged_data.parquet（由 merge_data.py 生成）
输出：data/processed/basic_cleaned.parquet
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

# 减少 pandas 操作中的不必要复制（pandas>=2.0 支持），大幅降低内存峰值
pd.set_option("mode.copy_on_write", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_PROCESSED, NAMECHANGE_PATH, STOCK_BASIC_PATH

# 指数代码前缀（上证指数 SH000xxx、深证指数 SZ399xxx），从股票池剔除
INDEX_PREFIXES = ("SH000", "SZ399")


def filter_index_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """剔除指数代码（SH000xxx 上证指数、SZ399xxx 深证指数），防止混入股票池"""
    before = len(df)
    df = df[~df["symbol"].str.startswith(INDEX_PREFIXES)]
    after = len(df)
    print(f"   - 剔除指数代码（SH000/SZ399）: {before - after} 行")
    return df


def load_stock_basic(df: pd.DataFrame) -> pd.DataFrame:
    """加载证券静态信息表并 merge list_date 列（无表时警告跳过）"""
    if not STOCK_BASIC_PATH.exists():
        print(f"   ⚠️ 未找到 {STOCK_BASIC_PATH.name}，跳过 list_date 合并（次新股过滤不可用）")
        return df
    basic = pd.read_parquet(STOCK_BASIC_PATH, columns=["symbol", "list_date"])
    df = df.merge(basic, on="symbol", how="left")
    print(f"   - 已合并 list_date（{basic['symbol'].nunique()} 只股票有上市日期）")
    return df


def filter_st_by_namechange(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 tushare namechange 区间表做 Point-in-Time ST/退市股过滤。

    对每个 (symbol, date)，取 start_date <= date 的最近一条名称记录；
    若该记录 end_date 早于 date（已失效）则视为无生效名称，不剔除；
    若生效名称含 ST/退 则剔除。用历史名称判断，避免用当前名称误判历史状态。
    """
    if not NAMECHANGE_PATH.exists():
        print(f"   ⚠️ 未找到 {NAMECHANGE_PATH.name}，跳过 Point-in-Time ST 过滤")
        return df
    nc = pd.read_parquet(NAMECHANGE_PATH)
    nc = nc.dropna(subset=["start_date"]).sort_values("start_date")

    # merge_asof 要求 on 键（date）全局单调、by 键类型一致：统一转字符串（df 的 symbol 可能是 category）
    nc["symbol"] = nc["symbol"].astype(str)

    # 只取需要的列做匹配，控制内存；left 只按 date 排序（merge_asof 要求 on 键全局单调）
    sub = df[["symbol", "date"]].reset_index()
    sub = sub.rename(columns={"index": "row_id"})
    sub["symbol"] = sub["symbol"].astype(str)
    sub = sub.sort_values("date")

    matched = pd.merge_asof(
        sub, nc,
        left_on="date", right_on="start_date",
        by="symbol", direction="backward",
    ).set_index("row_id")

    # 区间失效检查：end_date 非空且早于当前日期 -> 该记录已失效
    valid = matched["end_date"].isna() | (matched["date"] <= matched["end_date"])
    is_st = matched["name"].str.contains("ST|退", na=False, case=False) & valid
    is_st = is_st.reindex(df.index).fillna(False)

    before = len(df)
    df = df[~is_st]
    after = len(df)
    print(f"   - 剔除 ST/退市股（Point-in-Time）: {before - after} 行")
    return df


def filter_st_stocks(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """剔除名称中包含 ST、*ST、退 的股票（若 name 列不存在则跳过，兜底方案）"""
    if name_col not in df.columns:
        print("   ⚠️ 无 name 列，跳过 ST/退市股过滤（兜底方案）")
        return df

    # 如果不包含 ST 且不包含 退 则保留
    before = len(df)
    df = df[~df[name_col].str.contains("ST|退|\\*ST", na=False, case=False)]
    after = len(df)
    print(f"   - 剔除 ST/退市股: {before - after} 行")
    return df


def filter_new_stocks(
    df: pd.DataFrame,
    list_date_col: str = "list_date",
    min_days: int = 60,
) -> pd.DataFrame:
    """
    剔除上市不足 min_days 个交易日的次新股。
    需要数据中包含 list_date 列（上市日期，YYYYMMDD 格式或 datetime）
    """
    if list_date_col not in df.columns:
        print("   ⚠️ 无 list_date 列，跳过次新股过滤")
        return df

    # 确保 list_date 是 datetime 类型
    df[list_date_col] = pd.to_datetime(df[list_date_col])

    # 计算上市天数（交易日近似，直接用自然日差值）
    df["days_since_list"] = (df["date"] - df[list_date_col]).dt.days

    before = len(df)
    df = df[df["days_since_list"] >= min_days]
    after = len(df)
    print(f"   - 剔除上市不足 {min_days} 天次新股: {before - after} 行")

    # 删除临时列
    df = df.drop(columns=["days_since_list"])
    return df


# ---------------------------------------------------------------------------
# 涨跌停识别（一字板不可交易问题）
# ---------------------------------------------------------------------------
def limit_pct_vec(symbols: pd.Series, dates: pd.Series) -> pd.Series:
    """按板块与日期返回涨跌幅限制（小数），向量化。

    规则（ST 股已在清洗中剔除，无需考虑 5%）：
      - 科创板（SH68）：20%
      - 创业板（SZ30）：2020-08-24 注册制改革起 20%，此前 10%
      - 北交所（BJ）：30%
      - 主板（SH60 / SZ00 等）：10%
    """
    pct = pd.Series(0.10, index=symbols.index, dtype="float64")
    pct[symbols.str.startswith("SH68")] = 0.20
    gem_reform = symbols.str.startswith("SZ30") & (dates >= pd.Timestamp("2020-08-24"))
    pct[gem_reform] = 0.20
    pct[symbols.str.startswith("BJ")] = 0.30
    return pct


def mark_limit_up_down(df: pd.DataFrame) -> pd.DataFrame:
    """识别涨跌停与一字板（封死无法成交）并新增 4 列布尔标记。

    limit_up         收盘价达到涨停价（含一字 / T 字板）
    limit_down       收盘价达到跌停价
    lock_limit_up    一字涨停：open=high=low=close=涨停价（开盘即封死，无法买入）
    lock_limit_down  一字跌停（无法卖出，持仓无法离场）

    口径说明：
      - 使用前复权 close：非除权日即真实成交价，可直接与涨停价比较；
        除权日复权抹平缺口，不会把除权下跌误判为跌停（最多漏判当日涨停，可接受）。
      - 涨停价 = round(昨收 × (1 + 幅度), 2)；昨收 = 组内 close.shift(1)。
      - 上市首日 / 停牌占位行（volume 缺失）不判断（无法确认成交状态）。
      - 幅度由 limit_pct_vec 按板块 + 日期确定（创业板 2020-08-24 前为 10%）。
      - 一字板 = 全天 high 与 low 相等（无波动区间），且收盘封在涨/跌停价。
    """
    sym = df["symbol"].astype(str)
    traded = df["volume"].notna() & (df["volume"] > 0)
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    pct = limit_pct_vec(sym, df["date"])

    up_price = (prev_close * (1 + pct)).round(2)
    dn_price = (prev_close * (1 - pct)).round(2)

    has_prev = prev_close.notna() & (prev_close > 0)
    at_up = df["close"] >= up_price - 1e-6
    at_dn = df["close"] <= dn_price + 1e-6

    df["limit_up"] = (traded & has_prev & at_up).astype(bool)
    df["limit_down"] = (traded & has_prev & at_dn).astype(bool)

    # 一字板：high 与 low 相等（相对容差，兼容 float32 仙股精度）
    one_line = (df["high"] - df["low"]).abs() <= (df["close"].abs() * 1e-3 + 1e-4)
    df["lock_limit_up"] = (one_line & df["limit_up"]).astype(bool)
    df["lock_limit_down"] = (one_line & df["limit_down"]).astype(bool)
    return df


def main():
    parser = argparse.ArgumentParser(description="基础数据清洗")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PROCESSED / "merged_data.parquet",
        help=f"输入文件（默认: {DATA_PROCESSED / 'merged_data.parquet'}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PROCESSED / "basic_cleaned.parquet",
        help=f"输出文件（默认: {DATA_PROCESSED / 'basic_cleaned.parquet'}）",
    )
    parser.add_argument(
        "--no-filter-st",
        action="store_true",
        help="跳过 ST/退市股过滤",
    )
    parser.add_argument(
        "--no-filter-new",
        action="store_true",
        help="跳过次新股过滤",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧹 开始基础数据清洗...")
    print(f"   输入: {args.input}")
    print(f"   输出: {args.output}")
    print("=" * 60)

    # 1. 加载数据
    if not args.input.exists():
        print(f"[ERROR] 未找到输入文件: {args.input}")
        return 1

    df = pd.read_parquet(args.input)
    print(f"📊 加载数据: {len(df):,} 行, {len(df.columns)} 列")

    # 2. 去重：保留唯一的 (symbol, date)
    # 注意：数据整体按 date 升序（merge_asof 输出顺序），drop_duplicates 在近似有序数据上开销较低
    before = len(df)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="first")
    print(f"   - 去重: {before - len(df)} 行")

    # 3. 基础过滤（顺序：指数 -> ST/退市 -> 次新股）
    # 指数是行级过滤（不依赖外部表），始终执行
    df = filter_index_stocks(df)

    if not args.no_filter_st:
        # 优先 Point-in-Time（namechange 区间表，防未来函数）；无表时退回 name 列兜底
        if NAMECHANGE_PATH.exists():
            df = filter_st_by_namechange(df)
        else:
            df = filter_st_stocks(df)

    if not args.no_filter_new:
        # 先合并 list_date（证券静态信息表），再做次新股过滤
        if STOCK_BASIC_PATH.exists() and "list_date" not in df.columns:
            df = load_stock_basic(df)
        df = filter_new_stocks(df, min_days=60)

    # 4. 停牌日价格保留 NaN（不填充，对齐 Qlib 业界做法）
    # Qlib：停牌日 OHLCV = NaN，不做 FFill。原因：
    #   - FFill 会让停牌期间产生"收益为 0"的假象，复牌跳空被摊平进标签/特征；
    #   - 停牌股在回测中本不可交易，不应产生训练标签；
    #   - 特征 NaN 由 LightGBM 原生处理，标签 NaN 在下游丢弃。
    # 注：price_cols 仅用于第 5 步识别财务列，不再用于填充。
    price_cols = ["open", "high", "low", "close", "volume", "amount", "vwap", "adjclose", "factor"]
    print("   - 停牌日价格保留 NaN（不填充，对齐 Qlib：停牌 OHLCV=NaN）")

    # 5. 财务缺失值前向填充（FFill）：用距离今日最近的过去数据填充，防止未来函数
    # 说明：merge_asof 已按 ann_date 做 Point-in-Time 对齐（公告日 <= 当前日期的最近值），
    # 这里再按股票分组 ffill，把首个公告日之前的缺口也补上（上市初期无财报期），
    # 只使用过去信息，绝不使用未来数据。
    # 识别财务列：排除主键、日期、价格列
    exclude_cols = {"symbol", "date", "ann_date"} | set(price_cols)
    fin_cols = [c for c in df.columns if c not in exclude_cols]

    if fin_cols:
        # merge_asof 输出整体按 date 升序 → 每只股票组内日期天然升序，无需再排序
        # （仅当整体 date 非单调时兜底排序，避免大表重复排序的内存/时间开销）
        if not df["date"].is_monotonic_increasing:
            print("   ⚠️ date 非单调递增，兜底排序...")
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # 只对确实含缺失值的列做 ffill，并分批处理以控制峰值内存
        cols_with_na = [c for c in fin_cols if df[c].isna().any()]
        if not cols_with_na:
            print("   - 财务字段无缺失，跳过 FFill")
        else:
            before_missing = int(df[cols_with_na].isna().sum().sum())
            BATCH = 12  # 每批最多 12 列，避免一次性 groupby 全部列产生巨大临时数组
            for i in range(0, len(cols_with_na), BATCH):
                batch = cols_with_na[i : i + BATCH]
                df[batch] = df.groupby("symbol", sort=False, observed=True)[batch].ffill()
                gc.collect()
            after_missing = int(df[cols_with_na].isna().sum().sum())
            print(f"   - 财务缺失值前向填充（FFill，防未来函数）: {len(cols_with_na)} 个字段")
            print(f"     填充缺失值: {before_missing - after_missing:,} 个；"
                  f"剩余 {after_missing:,} 个为全程无财报的股票，保留 NaN（不再用 0 伪装）")

    # 6. 涨跌停 / 一字板标记（识别"无法成交"的封板日，供下游标签剔除与风控使用）
    print("   - 识别涨跌停 / 一字板...")
    df = mark_limit_up_down(df)
    n_up = int(df["limit_up"].sum())
    n_dn = int(df["limit_down"].sum())
    n_lock_up = int(df["lock_limit_up"].sum())
    n_lock_dn = int(df["lock_limit_down"].sum())
    print(f"   - 涨停 {n_up:,} 行（其中一字涨停 {n_lock_up:,}，无法买入）")
    print(f"   - 跌停 {n_dn:,} 行（其中一字跌停 {n_lock_dn:,}，无法卖出）")

    # 7. 内存优化
    print("💾 优化数据类型...")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object":
            df[col] = df[col].astype("category")
        elif col_type == "float64":
            df[col] = df[col].astype("float32")
        elif col_type == "int64":
            if not df[col].isna().any():
                df[col] = df[col].astype("int32")
            else:
                df[col] = df[col].astype("float32")
    gc.collect()

    # 7. 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\n✅ 基础清洗完成！")
    print(f"   输出文件: {args.output}")
    print(f"   最终行数: {len(df):,}")
    print(f"   最终列数: {len(df.columns)}")
    print(f"   文件大小: {args.output.stat().st_size / 1024 / 1024:.1f} MB")

    # 8. 快速摘要
    print("\n📋 数据摘要:")
    print(f"   - 日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"   - 股票数量: {df['symbol'].nunique()}")
    print(f"   - 财务字段数量: {len(fin_cols)}")
    print(f"   - 涨跌停标记: limit_up / limit_down / lock_limit_up / lock_limit_down")

    return 0


if __name__ == "__main__":
    sys.exit(main())