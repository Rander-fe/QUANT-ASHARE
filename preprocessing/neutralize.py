# -*- coding: utf-8 -*-
"""环节：因子中性化（行业 / 市值 / 联合）。

职责：
    1. 加载自研因子宽表 factors.parquet（已含标签）
    2. 合并 extra 底表（basic_cleaned_with_extra_by_date.parquet）的
       industry（行业）、total_mv（总市值，万元）
    3. 按模式逐日横截面 OLS 回归取残差：
         industry : 因子 ~ 行业哑变量（去掉截距）
         market   : 因子 ~ log(总市值)
         both     : 因子 ~ 行业哑变量 + log(总市值)（默认）
    4. 输出中性化后因子表 + 训练集口径 IC 对比报告（原始 vs 中性化后）

输入：
    data/processed/factors.parquet
    data/processed/basic_cleaned_with_extra_by_date.parquet

输出：
    data/processed/neutralized_{mode}.parquet          中性化后因子（保留标签）
    data/processed/neutralized_{mode}_summary.parquet  IC 对比报告（训练集口径）

内存策略：
    factors.parquet 43 个 row group 按日期对齐（每块 60 个交易日），
    逐块读入 -> 逐日中性化 -> 逐块写回，峰值内存 = 单块（~16 万行 × 275 列）。

用法：
    python preprocessing/neutralize.py                    # 默认 both（行业+市值联合）
    python preprocessing/neutralize.py --mode industry    # 仅行业中性化
    python preprocessing/neutralize.py --mode market      # 仅市值中性化
    python preprocessing/neutralize.py --mode all         # 三种模式全部跑

说明：
    - 中性化核心复用 preprocessing/preprocess_factors.neutralize_cross_section；
    - 报告 IC 用训练集（TRAIN_PERIOD）日频 RankIC（Spearman），
      label 窗口 purge 规则与 remove_redundant_factors 一致（末尾 5 天）。
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BASIC_EXTRA_PATH,
    DATA_PROCESSED,
    TRAIN_PERIOD,
)
from preprocessing.preprocess_factors import (
    KEY_COLS,
    LABEL_COLS,
    LOG_MV_COL,
    neutralize_cross_section,
)

warnings.filterwarnings("ignore", category=UserWarning, message=r".*input array is constant.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*input array is constant.*")

FACTORS_FILE = "factors.parquet"
NEUTRAL_EPS = 1e-12

# 中性化模式 -> (行业列, 市值列)
MODES: dict[str, tuple[str | None, str | None]] = {
    "industry": ("industry", None),
    "market": (None, LOG_MV_COL),
    "both": ("industry", LOG_MV_COL),
}


def load_factors_pf() -> pq.ParquetFile:
    """打开 factors.parquet，返回 ParquetFile 句柄。"""
    path = DATA_PROCESSED / FACTORS_FILE
    if not path.exists():
        print(f"[ERROR] 缺少因子表: {path}")
        print("        请先运行: python main.py build_factors")
        sys.exit(1)
    return pq.ParquetFile(str(path))


def load_extra_mapping() -> tuple[pd.DataFrame, str | None, str | None]:
    """加载 extra 底表（行业 + log市值），返回 (extra, industry_col, mv_col)。

    extra 只保留 (symbol, date, industry, log_mv)，减小内存。
    """
    if not BASIC_EXTRA_PATH.exists():
        print(f"[WARN] 缺少 extra 底表: {BASIC_EXTRA_PATH}")
        print("       跳过中性化元数据加载（将无法中性化）")
        return pd.DataFrame(), None, None

    pf = pq.ParquetFile(str(BASIC_EXTRA_PATH))
    schema_names = pf.schema.names
    read_cols = ["symbol", "date", "industry", "total_mv"]
    read_cols = [c for c in read_cols if c in schema_names]
    extra = pf.read(columns=read_cols).to_pandas()
    extra["date"] = pd.to_datetime(extra["date"])

    industry_col = "industry" if "industry" in extra.columns else None
    mv_col = None
    if "total_mv" in extra.columns:
        extra["total_mv"] = pd.to_numeric(extra["total_mv"], errors="coerce")
        # tushare daily_basic 的 total_mv 单位为万元，转成元取对数
        mv = extra["total_mv"] * 1e4
        extra[LOG_MV_COL] = np.log(mv.clip(lower=1.0))
        extra = extra.drop(columns=["total_mv"])
        mv_col = LOG_MV_COL

    keep = ["symbol", "date"] + [c for c in (industry_col, mv_col) if c]
    extra = extra[keep]
    print(f"[INFO] extra 底表: {len(extra):,} 行, 行业列={industry_col}, 市值列={mv_col}")
    return extra, industry_col, mv_col


def neutralize_chunk(
    chunk: pd.DataFrame,
    extra: pd.DataFrame,
    feature_cols: list[str],
    ind_col: str | None,
    mv_col: str | None,
) -> pd.DataFrame:
    """处理一个 row group：合并 extra -> 逐日横截面中性化（取残差）。

    返回与 chunk 行序一致的 DataFrame，仅特征列被替换为残差。

    内存策略（与 preprocess_factors.process_chunk 一致）：
        直接操作传入的 chunk（调用方用后即弃，无需 copy），
        逐日取单日小副本（约 2700 行 × 275 列 ≈ 6MB）计算后
        「就地写回」chunk，不累积 parts、不做整块 concat——
        旧实现 parts 累积 + concat 会产生 3~4 份整块副本（每块 ~6GB），
        是上次 neutralized_both.parquet 写入中断/OOM 的根因。
    """
    out = chunk  # 就地操作，避免整块 copy

    if ind_col is None and mv_col is None:
        # 无元数据：原样返回（退化模式，仍可产出报告基线）
        return out

    # 合并行业/市值（chunk 中可能已有同名列，先剔除避免 _x/_y 冲突）
    meta_cols = [c for c in (ind_col, mv_col) if c]
    dup_meta = [c for c in meta_cols if c in out.columns]
    if dup_meta:
        out = out.drop(columns=dup_meta)

    # 性能优化：extra 已按日期排序，二分定位本块 [d_min, d_max] 范围再 merge，
    # 避免每块与全量 1000 万行 extra 做 left join（43 块 x 3 模式会极慢）。
    d_min = out["date"].min().to_datetime64()
    d_max = out["date"].max().to_datetime64()
    extra_dates = extra["date"].values  # datetime64[ns]
    lo = int(np.searchsorted(extra_dates, d_min, side="left"))
    hi = int(np.searchsorted(extra_dates, d_max, side="right"))
    extra_sub = extra.iloc[lo:hi]
    out = out.merge(extra_sub[["symbol", "date"] + meta_cols], on=KEY_COLS, how="left")

    # 仅市值模式（无行业列）：构造恒值行业占位列。
    # 单哑变量等价于截距，回归 Y ~ 1 + log_mv 取残差 = 去均值 + 去市值线性。
    dummy_ind: str | None = None
    if ind_col is None and mv_col is not None:
        dummy_ind = "__ALL__"
        out[dummy_ind] = "ALL"
        ind_col = dummy_ind

    # 统一升 float64 计算（避免残差写入 float32 列的 dtype 警告），
    # 逐日处理后就地写回，最后整体降回 float32 落盘。
    out[feature_cols] = out[feature_cols].astype(float)

    dates = out["date"].unique()
    for d in dates:
        mask = out["date"] == d
        day = out.loc[mask].copy()
        day = neutralize_cross_section(
            day, feature_cols, industry_col=ind_col, mv_col=mv_col
        )
        # 就地写回（用 .values 避免索引对齐开销）
        out.loc[mask, feature_cols] = day[feature_cols].values

    if dummy_ind is not None and dummy_ind in out.columns:
        out = out.drop(columns=[dummy_ind])

    out[feature_cols] = out[feature_cols].astype(np.float32)
    return out


def compute_daily_rankic(
    df: pd.DataFrame, factor_cols: list[str], label_col: str
) -> pd.DataFrame:
    """按交易日分组计算日频 RankIC（Spearman），返回 [date x factor]。"""
    daily_parts = []
    for date, group in df.groupby("date", sort=True):
        valid = group[label_col].notna()
        if valid.sum() < 5:
            continue
        sub = group.loc[valid]
        ic = sub[factor_cols].corrwith(sub[label_col], method="spearman")
        ic.name = date
        daily_parts.append(ic)
    if not daily_parts:
        return pd.DataFrame()
    return pd.DataFrame(daily_parts)


def summarize_ic(daily_ic: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """基于日频 IC 表计算汇总统计（IC 均值/标准差/ICIR），按 |IC| 降序。"""
    valid_factors = [c for c in factor_cols if c in daily_ic.columns]
    if not valid_factors:
        return pd.DataFrame(columns=["factor", "ic_mean", "ic_std", "ic_count", "icir"])
    d = daily_ic[valid_factors]
    summary = pd.DataFrame(
        {
            "ic_mean": d.mean(),
            "ic_std": d.std(),
            "ic_count": d.count(),
        }
    )
    summary["icir"] = summary["ic_mean"] / (summary["ic_std"] + NEUTRAL_EPS)
    summary = summary.reset_index().rename(columns={"index": "factor"})
    summary = summary.sort_values("ic_mean", ascending=False).reset_index(drop=True)
    return summary


def build_ic_report(
    daily_ic_raw: pd.DataFrame,
    daily_ic_neu: pd.DataFrame,
    factor_cols: list[str],
    mode: str,
    train_days: int,
) -> pd.DataFrame:
    """对比原始 vs 中性化后的 IC，生成报告表。

    列：factor / ic_mean_raw / icir_raw / ic_mean_neu / icir_neu /
        ic_change（中性化后 IC 变化）/ retention（保留率 = 中性化/原始，防除零）。
    """
    raw = summarize_ic(daily_ic_raw, factor_cols).set_index("factor")
    neu = summarize_ic(daily_ic_neu, factor_cols).set_index("factor")
    report = pd.DataFrame(index=factor_cols)
    report["ic_mean_raw"] = raw["ic_mean"]
    report["icir_raw"] = raw["icir"]
    report["ic_count_raw"] = raw["ic_count"]
    report["ic_mean_neu"] = neu["ic_mean"]
    report["icir_neu"] = neu["icir"]
    report["ic_count_neu"] = neu["ic_count"]
    report["ic_change"] = report["ic_mean_neu"] - report["ic_mean_raw"]
    report["retention"] = (
        report["ic_mean_neu"] / report["ic_mean_raw"].replace(0, np.nan)
    )
    report["mode"] = mode
    report["train_days"] = train_days
    report = report.reset_index().rename(columns={"index": "factor"})
    report = report.sort_values("ic_mean_raw", ascending=False).reset_index(drop=True)
    return report


def run_mode(mode: str) -> int:
    """运行一种中性化模式，返回退出码。"""
    ind_col, mv_col = MODES[mode]
    t0 = time.time()
    print("=" * 64)
    print(f"[INFO] 中性化模式 [{mode}]: 行业={ind_col or '-'}, 市值={mv_col or '-'}")
    print("=" * 64)

    pf = load_factors_pf()
    extra, _, _ = load_extra_mapping()
    # 按日期排序，供 neutralize_chunk 做范围切片（每块只 merge 对应 60 天）
    extra = extra.sort_values("date").reset_index(drop=True)

    schema_names = pf.schema.names
    label_cols = [c for c in LABEL_COLS if c in schema_names]
    if not label_cols:
        print("[ERROR] factors.parquet 无标签列")
        return 1
    label_col = label_cols[0]  # label_ret_5 优先（与 remove_redundant 口径一致）
    feature_cols = [c for c in schema_names if c not in KEY_COLS + label_cols]
    print(f"[INFO] {pf.metadata.num_rows:,} 行, {len(feature_cols)} 个特征, 标签={label_col}")

    # ---- 1. 训练集日频 IC（原始因子）----
    train_start, train_end = TRAIN_PERIOD
    n_rg = pf.metadata.num_row_groups
    raw_ic_parts: list[pd.DataFrame] = []
    neu_ic_parts: list[pd.DataFrame] = []

    out_path = DATA_PROCESSED / f"neutralized_{mode}.parquet"
    writer: pq.ParquetWriter | None = None

    try:
        for i in range(n_rg):
            chunk = pf.read_row_group(i).to_pandas()
            chunk["date"] = pd.to_datetime(chunk["date"])

            # 只处理训练集时间范围内的行（IC 口径：训练集，purge 末尾 label 窗口）
            mask_train = (chunk["date"] >= pd.Timestamp(train_start)) & (
                chunk["date"] <= pd.Timestamp(train_end)
            )
            train_chunk = chunk[mask_train].copy()

            # 原始 IC（训练集）
            if len(train_chunk):
                raw_ic = compute_daily_rankic(train_chunk, feature_cols, label_col)
                if len(raw_ic):
                    raw_ic_parts.append(raw_ic)

            # 中性化
            neu_chunk = neutralize_chunk(chunk, extra, feature_cols, ind_col, mv_col)
            if len(train_chunk):
                neu_train = neu_chunk[mask_train]
                neu_ic = compute_daily_rankic(neu_train, feature_cols, label_col)
                if len(neu_ic):
                    neu_ic_parts.append(neu_ic)

            # 写回（全样本）
            out_df = neu_chunk[KEY_COLS + label_cols + feature_cols]
            if writer is None:
                writer = pq.ParquetWriter(
                    str(out_path), schema=pa.Schema.from_pandas(out_df)
                )
            writer.write_table(pa.Table.from_pandas(out_df, preserve_index=False))

            del chunk, neu_chunk, train_chunk
            gc.collect()
            if (i + 1) % 10 == 0 or i == n_rg - 1:
                print(f"   [OK] row group {i + 1}/{n_rg}（{time.time() - t0:.0f}s）")
    finally:
        if writer is not None:
            writer.close()

    # ---- 2. 合并日频 IC 并 purge ----
    raw_daily = pd.concat(raw_ic_parts, axis=0).sort_index()
    neu_daily = pd.concat(neu_ic_parts, axis=0).sort_index()
    # purge：训练集末尾 label 窗口（5 个交易日）的 IC 标签伸入验证集，剔除
    purge_days = 5
    raw_daily = raw_daily.iloc[:-purge_days] if len(raw_daily) > purge_days else raw_daily
    neu_daily = neu_daily.iloc[:-purge_days] if len(neu_daily) > purge_days else neu_daily

    # ---- 3. 报告 ----
    report = build_ic_report(raw_daily, neu_daily, feature_cols, mode, len(raw_daily))
    report_out = DATA_PROCESSED / f"neutralized_{mode}_summary.parquet"
    report.to_parquet(report_out, index=False)

    # ---- 4. 控制台摘要 ----
    n_better = (report["ic_change"] > 0).sum()
    n_worse = (report["ic_change"] < 0).sum()
    mean_retention = report["retention"].abs().mean()
    print("\n[REPORT] 中性化前后 IC 对比（训练集口径，Top 10）:")
    cols = ["factor", "ic_mean_raw", "icir_raw", "ic_mean_neu", "icir_neu", "retention"]
    print(report.head(10)[cols].to_string(index=False, float_format="%.4f"))
    print(f"\n   IC 提升因子数: {n_better} / 不变: {len(report) - n_better - n_worse} / 下降: {n_worse}")
    print(f"   平均保留率 |IC_neu/IC_raw|: {mean_retention:.3f}（>0.5 说明因子未被过度中性化）")
    print(f"\n[DONE] 中性化完成 [{mode}]:")
    print(f"   因子表: {out_path}")
    print(f"   报告: {report_out}")
    print(f"   耗时: {time.time() - t0:.0f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="因子中性化：行业/市值/联合")
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["industry", "market", "both", "all"],
        help="中性化模式：industry=仅行业, market=仅市值, both=联合（默认）, all=全跑",
    )
    args = parser.parse_args()

    if args.mode == "all":
        codes = [run_mode(m) for m in MODES]
        return 1 if any(c != 0 for c in codes) else 0
    return run_mode(args.mode)


if __name__ == "__main__":
    sys.exit(main())
