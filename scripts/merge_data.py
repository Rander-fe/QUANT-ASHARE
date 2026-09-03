# -*- coding: utf-8 -*-
"""
数据合并与清洗 —— 将日线行情 + 财务事件表合并为一张大宽表。

处理流程：
    1. 加载并合并日线数据（SH + SZ，忽略北交所）
    2. 加载财务事件表（Point-in-Time 快照）
    3. 使用 pd.merge_asof 按 (symbol, date) 精确对齐
    4. 处理缺失值（停牌日价格保留 NaN 不填充，对齐 Qlib；财务保持 Point-in-Time）
    5. 保存为 parquet 供后续因子构建使用

输出：data/processed/merged_data.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DAILY_RAW_DIR, FIN_PROCESSED_DIR, DATA_PROCESSED

def main():
    # 1. 创建输出目录
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📊 开始合并日线数据与财务数据...")
    print("=" * 60)

    # 2. 加载日线数据（仅沪市 + 深市，北交所暂时忽略）
    print("\n📈 加载日线数据...")
    df_daily_list = []
    for prefix in ["sh", "sz"]:
        fpath = DAILY_RAW_DIR / f"daily_{prefix}.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            print(f"   - {fpath.name}: {len(df):,} 行")
            df_daily_list.append(df)

    if not df_daily_list:
        print("[ERROR] 未找到日线数据")
        return 1

    df_daily = pd.concat(df_daily_list, ignore_index=True)

    # 确保日期格式一致
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"✅ 日线数据汇总：{len(df_daily):,} 行，{df_daily['symbol'].nunique()} 只股票")

    # 3. 加载财务事件表
    print("\n📊 加载财务事件表...")
    fin_path = FIN_PROCESSED_DIR / "financial_events.parquet"
    if not fin_path.exists():
        print(f"[ERROR] 未找到 {fin_path}，请先运行 fetch_financial.py")
        return 1

    df_fin = pd.read_parquet(fin_path)
    # 确保 ann_date 是日期类型
    df_fin["ann_date"] = pd.to_datetime(df_fin["ann_date"])
    df_fin = df_fin.sort_values(["symbol", "ann_date"]).reset_index(drop=True)
    print(f"✅ 财务事件表：{len(df_fin):,} 行，{df_fin['symbol'].nunique()} 只股票")

    # 4. 关键步骤：Point-in-Time 合并（使用 merge_asof 防止未来函数）
    print("\n🔗 开始合并（按日期向后匹配，取最近公告）...")
    # 注意：merge_asof 要求两个表都按合并键（left_on / right_on）排序
    # 使用 'backward' 方向：取公告日 <= 当前日期的最近一条记录
    df_merged = pd.merge_asof(
        df_daily.sort_values("date"),          # 左表：日线数据
        df_fin.sort_values("ann_date"),        # 右表：财务事件
        left_on="date",                        # 左表匹配列
        right_on="ann_date",                   # 右表匹配列
        by="symbol",                           # 按股票分组匹配
        direction="backward",                  # 取过去的最近公告
    )

    print(f"✅ 合并完成：{len(df_merged):,} 行，{df_merged['symbol'].nunique()} 只股票")

    # 5. 处理合并后缺失值（停牌/无财报情况）
    print("\n🧹 处理缺失值...")
    
    # 5.1 日线字段：停牌日价格保留 NaN（不填充，对齐 Qlib 业界做法）
    # Qlib：停牌日 OHLCV = NaN，不做 FFill。原因：
    #   - FFill 会把停牌期间"收益为 0"的假象带入标签/特征（复牌跳空被摊平）；
    #   - 停牌股在回测中本不可交易，不应产生训练标签；
    #   - 特征 NaN 由 LightGBM 原生处理，标签 NaN 在下游丢弃。
    # 注：cols_to_ffill 仅用于第 8 步摘要识别财务字段，不再做填充。
    cols_to_ffill = ["open", "high", "low", "close", "volume", "amount", "vwap", "adjclose"]

    # 5.2 财务字段：财报天然稀疏，保留 NaN，后续因子计算时可根据日期窗口决定是否填充
    # 也可以做简单的前向填充，但需谨慎（财报过期后不应再有效）
    # 建议：若后续因子计算需要，可自行在因子层决定处理方式，此处保持原样

    # 6. 检查内存占用并优化数据类型
    print("\n💾 优化数据类型以节省内存...")
    for col in df_merged.columns:
        col_type = df_merged[col].dtype
        if col_type == "object":
            # 字符串列转为 category 类型以节省内存
            df_merged[col] = df_merged[col].astype("category")
        elif col_type == "float64":
            # 浮点列转为 float32（精度足够）
            df_merged[col] = df_merged[col].astype("float32")
        elif col_type == "int64":
            # 整型列检查是否可转为 int32
            if not df_merged[col].isna().any():
                df_merged[col] = df_merged[col].astype("int32")
            else:
                df_merged[col] = df_merged[col].astype("float32")

    # 7. 保存结果
    out_path = DATA_PROCESSED / "merged_data.parquet"
    df_merged.to_parquet(out_path, index=False)
    print(f"\n✅ 大宽表已保存：{out_path}")
    print(f"   总行数：{len(df_merged):,}")
    print(f"   总列数：{len(df_merged.columns)}")
    print(f"   文件大小：{out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 8. 快速摘要
    print("\n📋 数据摘要：")
    print(f"   - 日期范围：{df_merged['date'].min()} ~ {df_merged['date'].max()}")
    print(f"   - 股票数量：{df_merged['symbol'].nunique()}")
    print(f"   - 财务字段：{len([c for c in df_merged.columns if c not in ['symbol', 'date', 'ann_date'] + cols_to_ffill])} 个")

    return 0

if __name__ == "__main__":
    sys.exit(main())