# -*- coding: utf-8 -*-
"""环节：因子预处理（去极值 + 行业/市值中性化 + 标准化）。

职责：
    1. 加载自研因子宽表 factors.parquet（已含全部 158 个 Alpha158 因子）
    2. 合并 extra 底表（basic_cleaned_with_extra_by_date.parquet）的
       industry（行业）、total_mv / circ_mv（市值）、涨跌停标记
    3. 每日横截面去极值（Winsorize，1%/99%）
    4. 每日横截面行业/市值中性化（OLS 回归取残差）
    5. 每日横截面 Z-Score 标准化
    6. 输出 features.parquet（中性化后特征）与 labels.parquet（原值标签）

输入：
    data/processed/factors.parquet
    data/processed/basic_cleaned_with_extra_by_date.parquet（行业/市值/涨跌停）

输出：
    data/processed/features.parquet   中性化+标准化后特征（含涨跌停标记列）
    data/processed/labels.parquet     标签（label_ret_5/10/20，原值）

内存策略：
    factors.parquet 为 43 个 row group、1042 万行 × 275 列；
    按 row group 流式读入，每次只处理一个分块（约 16 万行），
    逐日横截面计算后写回分块，最后统一落盘，峰值内存可控。

用法：
    python preprocessing/preprocess_factors.py [--quantiles 0.01 0.99]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BASIC_EXTRA_PATH,
    DATA_PROCESSED,
    FEATURES_FILE,
    LABELS_FILE,
)

# 主键列与标签列
KEY_COLS = ["symbol", "date"]
LABEL_COLS = ("label_ret_5", "label_ret_10", "label_ret_20")

# 从 extra 表合并的非特征列（中性化输入 + 交易状态标记）
EXTRA_COLS = ["industry", "total_mv", "circ_mv", "pct_chg"]
# 涨跌停标记列（重跑 clean_data 后才有，容错缺失）
LIMIT_COLS = ["limit_up", "limit_down", "lock_limit_up", "lock_limit_down"]

# 对数市值（中性化用市值对数，降低量纲影响）
LOG_MV_COL = "log_mv"

# 中小盘样本少，用按行业聚合的市值中位数兜底，避免单行业市值全 NaN 时中性化崩溃
_NEUTRAL_EPS = 1e-12


def load_factors_schema() -> pq.ParquetFile:
    """打开 factors.parquet，返回 ParquetFile 句柄。"""
    path = DATA_PROCESSED / "factors.parquet"
    if not path.exists():
        print(f"[ERROR] 缺少因子表: {path}")
        print("        请先运行: python main.py build_factors")
        sys.exit(1)
    return pq.ParquetFile(str(path))


def load_extra_mapping() -> tuple[pd.DataFrame, list[str]]:
    """加载 extra 底表并切片为 (symbol, date, 行业/市值) 长表。

    返回 (extra, meta_cols)。extra 只保留主键 + 非特征列，减小内存；
    meta_cols 为实际存在的非特征列（行业/市值/涨跌停）。
    """
    if not BASIC_EXTRA_PATH.exists():
        print(f"[WARN] 缺少 extra 底表: {BASIC_EXTRA_PATH}")
        print("       跳过行业/市值中性化（退化：仅去极值+标准化）")
        return pd.DataFrame(), []

    # 只读需要的列，避免加载整表（1 亿列 × 1000 万行的开销）
    cols = KEY_COLS + EXTRA_COLS + LIMIT_COLS
    pf = pq.ParquetFile(str(BASIC_EXTRA_PATH))
    schema_names = pf.schema.names
    read_cols = [c for c in cols if c in schema_names]
    extra = pf.read(columns=read_cols).to_pandas()

    # 统一 date 类型
    extra["date"] = pd.to_datetime(extra["date"])
    # 市值单位：tushare daily_basic 的 total_mv/circ_mv 单位为万元，转成元取对数
    if "total_mv" in extra.columns:
        extra["total_mv"] = pd.to_numeric(extra["total_mv"], errors="coerce")
        mv = extra["total_mv"] * 1e4  # 万元 -> 元
        extra[LOG_MV_COL] = np.log(mv.clip(lower=1.0))
        extra = extra.drop(columns=["total_mv"])

    if "circ_mv" in extra.columns:
        extra = extra.drop(columns=["circ_mv"])

    # 涨跌停列缺失（clean_data 未重跑）时补空列，保持接口一致
    for c in LIMIT_COLS:
        if c not in extra.columns:
            extra[c] = False

    meta_cols = [c for c in EXTRA_COLS + LIMIT_COLS + [LOG_MV_COL] if c in extra.columns]
    print(f"📊 extra 底表: {len(extra):,} 行, 非特征列: {meta_cols}")
    return extra, meta_cols


def winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    """横截面缩尾：按分位数截断（NaN 保留）。"""
    q_lo, q_hi = s.quantile([lower, upper]).iloc[[0, 1]]
    return s.clip(lower=q_lo, upper=q_hi)


def mark_limit_from_pct_chg(df: pd.DataFrame) -> pd.DataFrame:
    """用 tushare 官方 pct_chg（%）按板块/日期阈值生成 limit_up/down，覆盖旧口径。

    背景：clean_data 的 mark_limit_up_down 用前复权 close 与涨停价比较，
    除权日复权会抹平缺口导致漏判；tushare daily_basic 的 pct_chg 为
    不复权官方涨跌幅（百分比），口径更准，故以此为准覆盖。

    阈值（容差 0.5%，pct_chg 保留两位小数，涨停日显示 9.97~10.03）：
      - 科创板（SH68）：20%
      - 创业板（SZ30）：2020-08-24 注册制改革起 20%，此前 10%
      - 北交所（BJ）：30%
      - 主板（SH60 / SZ00 等）：10%
    ST 股已在 clean_data 剔除，无需考虑 5%。

    仅当 pct_chg 有效（非 NaN）时标记；缺失（停牌/新股/未拉到）保持 False。
    lock_limit 列依赖 high/low（preprocess 阶段已无此列），保留 extra 表原值。
    """
    sym = df["symbol"].astype(str)
    pct = pd.Series(10.0, index=df.index, dtype="float64")
    pct[sym.str.startswith("SH68")] = 20.0
    gem_reform = sym.str.startswith("SZ30") & (
        df["date"] >= pd.Timestamp("2020-08-24")
    )
    pct[gem_reform] = 20.0
    pct[sym.str.startswith("BJ")] = 30.0

    chg = pd.to_numeric(df.get("pct_chg"), errors="coerce")
    has_chg = chg.notna()
    df["limit_up"] = (has_chg & (chg >= pct - 0.5)).astype(bool)
    df["limit_down"] = (has_chg & (chg <= -(pct - 0.5))).astype(bool)
    return df


def neutralize_cross_section(
    df: pd.DataFrame,
    feature_cols: list[str],
    industry_col: str,
    mv_col: str | None,
) -> pd.DataFrame:
    """单日横截面中性化：因子 ~ 行业哑变量 + log(市值)，取残差。

    行业哑变量直接参与 OLS（去掉截距）；市值对数作为连续变量，
    为 None 时仅做行业中性化。有效样本不足时回退为原始值。
    """
    valid = df[industry_col].notna()
    if mv_col is not None:
        valid = valid & df[mv_col].notna() & (df[mv_col].abs() > 0)
    if valid.sum() < 30:
        # 有效样本太少，无法稳健估计，保留原值
        return df

    sub = df.loc[valid].copy()
    # 统一升 float64 计算，避免 float64 残差写入 float32 列的 dtype 警告
    sub[feature_cols] = sub[feature_cols].astype(float)

    X = pd.get_dummies(sub[industry_col], prefix="ind", dtype=float)
    if mv_col is not None:
        X[mv_col] = sub[mv_col].astype(float)

    # 按缺失模式分组，同一组的特征共享一次 OLS 求解（X 相同）：
    # 270 特征 × 逐日 的逐特征 lstsq（~70 万次）降为每天几次矩阵求解，提速数十倍。
    groups: dict[bytes, list[str]] = {}
    for col in feature_cols:
        m = sub[col].notna().values
        if m.sum() < 30:
            continue
        groups.setdefault(m.tobytes(), []).append(col)

    for mask_key, cols in groups.items():
        mask = np.frombuffer(mask_key, dtype=np.bool_)
        try:
            Xm = X[mask].values
            Ym = sub.loc[mask, cols].values
            # 最小二乘残差：Y - X@B（多列一次求解）
            beta, _, _, _ = np.linalg.lstsq(Xm, Ym, rcond=None)
            resid = Ym - Xm @ beta
            sub.loc[mask, cols] = resid
        except np.linalg.LinAlgError:
            continue

    # 合并回原 df（含 NaN 缺失样本），dtype 一致无警告
    df.loc[valid, feature_cols] = sub[feature_cols]
    return df


def process_chunk(
    chunk: pd.DataFrame,
    extra: pd.DataFrame,
    meta_cols: list[str],
    feature_cols: list[str],
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """处理一个 row group 分块：合并 extra -> 逐日去极值/中性化/标准化。

    分块内日期连续（factors.parquet 按日期排序），直接 merge extra。
    """
    # 1. 合并行业/市值/涨跌停（先剔除 chunk 中可能已存在的 meta 列，避免 _x/_y 冲突）
    if not extra.empty and meta_cols:
        dup_meta = [c for c in meta_cols if c in chunk.columns]
        if dup_meta:
            chunk = chunk.drop(columns=dup_meta)
        chunk = chunk.merge(
            extra[KEY_COLS + meta_cols], on=KEY_COLS, how="left"
        )

    # 1.5 官方口径涨跌停标记：用 pct_chg 覆盖 clean_data 的前复权 close 口径
    if "pct_chg" in chunk.columns:
        chunk = mark_limit_from_pct_chg(chunk)

    # 2. 统一升 float64 计算（避免部分赋值 dtype 警告），落盘前再降回 float32。
    #    注意：astype 会新分配一份 float64（整块约 6GB），旧 float32 随即被 GC，
    #    峰值仅短暂翻倍，可接受；真正的内存杀手是下面的逐日累积+concat，已消除。
    chunk[feature_cols] = chunk[feature_cols].astype(float)

    dates = chunk["date"].unique()

    # 3. 逐日处理：单日副本（约 2700 行 × 270 列 ≈ 6MB）很小，处理完就地写回 chunk，
    #    避免 processed_parts 累积整块副本 + pd.concat 整块拷贝（OOM 根因）。
    for d in dates:
        mask = chunk["date"] == d
        day = chunk.loc[mask].copy()

        # 3.1 去极值（Winsorize）：按列向量化一次完成
        q = day[feature_cols].quantile([lower, upper])
        day[feature_cols] = day[feature_cols].clip(
            lower=q.iloc[0], upper=q.iloc[1], axis=1
        )

        # 3.2 中性化（行业 + log市值）；无行业/市值元数据则跳过
        if not extra.empty and meta_cols and "industry" in meta_cols:
            mv_col = LOG_MV_COL if LOG_MV_COL in meta_cols else None
            day = neutralize_cross_section(day, feature_cols, "industry", mv_col)

        # 3.3 标准化（Z-Score）：向量化
        mean = day[feature_cols].mean()
        std = day[feature_cols].std()
        day[feature_cols] = (day[feature_cols] - mean) / (std + _NEUTRAL_EPS)
        # 常数列（std≈0）置 0，避免噪声放大
        const_cols = std[std < _NEUTRAL_EPS].index
        if len(const_cols):
            day[const_cols] = 0.0

        # 就地写回（用 .values 避免索引对齐开销）
        chunk.loc[mask, feature_cols] = day[feature_cols].values

    # 4. 落盘前降回 float32，节省存储
    chunk[feature_cols] = chunk[feature_cols].astype(np.float32)
    return chunk


def main() -> int:
    parser = argparse.ArgumentParser(description="因子预处理：去极值+中性化+标准化")
    parser.add_argument(
        "--quantiles", type=float, nargs=2, default=(0.01, 0.99),
        metavar=("Q_LO", "Q_HI"), help="Winsorize 分位数（默认 0.01 0.99）",
    )
    parser.add_argument(
        "--max-rg", type=int, default=None,
        help="调试用：只处理前 N 个 row group（验证内存后正式跑可不加）",
    )
    args = parser.parse_args()
    lower, upper = args.quantiles

    t0 = time.time()
    print("=" * 60)
    print("🧪 因子预处理：去极值 + 行业/市值中性化 + 标准化")
    print("=" * 60)

    # 1. 加载输入
    pf = load_factors_schema()
    extra, meta_cols = load_extra_mapping()

    schema_names = pf.schema.names
    label_cols = [c for c in LABEL_COLS if c in schema_names]
    feature_cols = [c for c in schema_names if c not in KEY_COLS + label_cols]
    print(f"📊 factors.parquet: {pf.metadata.num_rows:,} 行, {len(feature_cols)} 个特征, "
          f"{len(label_cols)} 个标签")

    # 2. 分块处理并落盘
    out_features = DATA_PROCESSED / FEATURES_FILE
    out_labels = DATA_PROCESSED / LABELS_FILE

    n_rg = pf.metadata.num_row_groups
    if args.max_rg is not None:
        n_rg = min(n_rg, args.max_rg)
        print(f"🔧 调试模式：仅处理前 {n_rg} 个 row group")
    label_frames: list[pd.DataFrame] = []
    total_rows = 0
    out_cols: list[str] = []

    # 用 pyarrow ParquetWriter 流式写多块（pandas to_parquet 的 mode=
    # 参数在 pyarrow>=14 已移除，改为显式 writer 逐块 append）
    writer: pq.ParquetWriter | None = None
    try:
        for i in range(n_rg):
            chunk = pf.read_row_group(i).to_pandas()
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk = process_chunk(chunk, extra, meta_cols, feature_cols, lower, upper)

            # 标签单独落盘（保留原值）
            lab = chunk[KEY_COLS + label_cols].copy()
            label_frames.append(lab)
            total_rows += len(chunk)

            # 特征落盘（保留额外元数据列）
            out_cols = KEY_COLS + [c for c in meta_cols if c in chunk.columns] + feature_cols
            # pct_chg 仅用于生成涨跌停标记，不作为特征落盘（防被误当特征/泄漏）
            out_cols = [c for c in out_cols if c != "pct_chg"]
            out_df = chunk[out_cols]
            if writer is None:
                writer = pq.ParquetWriter(
                    str(out_features), schema=pa.Schema.from_pandas(out_df)
                )
            writer.write_table(pa.Table.from_pandas(out_df, preserve_index=False))

            if (i + 1) % 10 == 0 or i == n_rg - 1:
                print(f"   ✅ 已处理 row group {i + 1}/{n_rg}（{time.time() - t0:.0f}s）")
    finally:
        if writer is not None:
            writer.close()

    # 3. 标签落盘
    labels = pd.concat(label_frames, ignore_index=True)
    labels.to_parquet(out_labels, index=False)

    print(f"\n✅ 因子预处理完成！")
    print(f"   特征: {out_features} ({total_rows:,} 行 × {len(out_cols)} 列)")
    print(f"   标签: {out_labels} ({len(labels):,} 行)")
    print(f"   耗时: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
