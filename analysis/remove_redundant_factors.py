# -*- coding: utf-8 -*-
"""
因子冗余剔除（基于日频 IC 序列相关性）

原理：
    冗余 = 两个因子在选股逻辑上干同一件事。判断标准不是公式长相，
    而是日频 RankIC 序列的相关性：若两个因子的 IC 每天都同涨同跌
    （相关 > 0.7），说明它们捕捉的是同一信息，保留 IC 更高的那个即可。

输入：
    data/processed/daily_rankic.parquet     评估产出的日频 IC（日期 × 因子）
    data/processed/factor_evaluation.parquet 评估产出的 IC 汇总（含 ic_mean）

输出：
    data/processed/selected_factor_cols.json   最终保留的因子列清单（供训练环节直接读取）
    data/processed/factor_kept.parquet         保留因子及指标
    data/processed/factor_removed.parquet      剔除因子、与谁冗余、相关性
    reports/factor_redundancy_report.txt       人类可读报告

算法：
    1. 按 ic_mean 降序排列因子（IC 越高优先级越高）
    2. 贪心遍历：每个因子若与「已保留集合」中任一因子相关 > 阈值 → 剔除，
       否则保留
    3. 剔除时记录与谁冗余（相关性最高的那个）、该因子自身的 IC

用法：
    python analysis/remove_redundant_factors.py [相关性阈值]
    默认阈值 0.70（可传 0.8 / 0.9 等覆盖）
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, REPORTS_DIR

DEFAULT_THRESHOLD = 0.70


def load_inputs(input_prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载日频 IC 与 IC 汇总表。"""
    stem = f"{input_prefix}_" if input_prefix else ""
    ic_path = DATA_PROCESSED / f"{stem}daily_rankic.parquet"
    ev_path = DATA_PROCESSED / f"{stem}factor_evaluation.parquet"

    if not ic_path.exists() or not ev_path.exists():
        print(f"[ERROR] 缺少输入，请先运行 analysis/evaluate_factors.py")
        print(f"        {ic_path} / {ev_path}")
        sys.exit(1)

    daily_ic = pd.read_parquet(ic_path)
    summary = pd.read_parquet(ev_path)
    return daily_ic, summary


def filter_by_abs_ic(
    summary: pd.DataFrame,
    min_abs_ic: float = 0.01,
    min_icir: float = 0.05,
) -> tuple[pd.DataFrame, int, int]:
    """按 |IC 均值| 与 |ICIR| 下限过滤因子。

    保留条件: |ic_mean| >= min_abs_ic 且 |icir| > min_icir。
    返回 (过滤后的 summary, 被 |IC| 过滤数, 被 |ICIR| 过滤数)。
    低 |IC| / 低 |ICIR| 因子信号≈噪声或不稳定，提前剔除减少后续计算。
    负 IC 因子若 |ICIR| 高（稳定负相关）仍保留——LightGBM 可学负向关系。
    """
    n_before = len(summary)
    mask_ic = summary["ic_mean"].abs() >= min_abs_ic
    mask_icir = summary["icir"].abs() > min_icir
    mask = mask_ic & mask_icir
    filtered = summary[mask].copy()
    n_ic_removed = int((~mask_ic).sum())
    n_icir_removed = int((mask_ic & ~mask_icir).sum())
    assert n_before == len(filtered) + n_ic_removed + n_icir_removed
    return filtered, n_ic_removed, n_icir_removed


def greedy_prune(
    daily_ic: pd.DataFrame,
    summary: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """贪心剔除：按 IC 降序保留，剔除与已保留因子相关 > 阈值者。

    返回 (kept, removed)：kept 为保留因子表，removed 为剔除记录表。
    """
    # 按 ic_mean 降序，确定优先级
    summary = summary.sort_values("ic_mean", ascending=False).reset_index(drop=True)
    factor_order = summary["factor"].tolist()

    # 因子间相关矩阵（日频 IC 序列，Pearson）
    corr = daily_ic[factor_order].corr()
    corr = corr.where(~np.eye(len(factor_order), dtype=bool))  # 对角线置 NaN

    kept: list[str] = []
    removed_rows: list[dict] = []

    for f in factor_order:
        if not kept:
            kept.append(f)
            continue
        # 与已保留因子的相关性
        sims = corr.loc[f, kept].dropna()
        if sims.empty:
            kept.append(f)
            continue
        max_sim = sims.max()
        if max_sim > threshold:
            # 与谁最冗余（相关性最高的保留因子）
            redundant_with = sims.idxmax()
            f_ic = summary.loc[summary["factor"] == f, "ic_mean"].iloc[0]
            keep_ic = summary.loc[summary["factor"] == redundant_with, "ic_mean"].iloc[0]
            removed_rows.append(
                {
                    "factor": f,
                    "ic_mean": f_ic,
                    "redundant_with": redundant_with,
                    "redundant_with_ic": keep_ic,
                    "corr": max_sim,
                }
            )
        else:
            kept.append(f)

    kept_df = summary[summary["factor"].isin(kept)].copy()
    removed_df = pd.DataFrame(removed_rows)
    return kept_df, removed_df


def build_funnel(
    total: int,
    n_filtered_ic: int,
    n_filtered_icir: int,
    n_kept: int,
    n_removed_redundant: int,
    min_abs_ic: float = 0.01,
    min_icir: float = 0.05,
    threshold: float = 0.70,
) -> list[str]:
    """各环节漏斗统计：每环节 过滤数 + 剩余数。返回文本行列表。"""
    n_after_ic = total - n_filtered_ic
    n_after_icir = n_after_ic - n_filtered_icir
    n_after_redundant = n_after_icir - n_removed_redundant
    lines = [
        "各环节漏斗统计:",
        "-" * 70,
        f"  {'环节':<28} {'过滤数':>8} {'剩余数':>8}",
        "-" * 70,
        f"  ① 初始因子                            {'-':>8} {total:>8}",
        f"  ② |IC| < {min_abs_ic} 过滤（低信号）   {n_filtered_ic:>8} {n_after_ic:>8}",
        f"  ③ |ICIR| < {min_icir} 过滤（不稳定）   {n_filtered_icir:>8} {n_after_icir:>8}",
        f"  ④ 冗余剔除（相关 > {threshold}）   {n_removed_redundant:>8} {n_after_redundant:>8}",
        f"  ⑤ 最终保留                             {'-':>8} {n_kept:>8}",
    ]
    return lines


def save_results(
    kept_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    threshold: float,
    total: int,
    n_filtered_ic: int = 0,
    n_filtered_icir: int = 0,
    min_abs_ic: float = 0.01,
    min_icir: float = 0.05,
    output_suffix: str = "",
    label: str | None = None,
) -> None:
    """落盘：因子清单 json + kept/removed parquet + 报告 txt。"""
    n_filtered = n_filtered_ic + n_filtered_icir
    # 1. 保留因子列清单（json，供训练环节读列）
    kept_cols = kept_df["factor"].tolist()
    suffix = f"_{output_suffix}" if output_suffix else ""
    json_path = DATA_PROCESSED / f"selected_factor_cols{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": threshold,
                "min_abs_ic": min_abs_ic,
                "min_icir": min_icir,
                "label": label,
                "n_total": total,
                "n_filtered_by_ic": n_filtered_ic,
                "n_filtered_by_icir": n_filtered_icir,
                "factors": kept_cols,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ 保留因子清单已保存: {json_path}")

    # 2. kept / removed 明细
    kept_path = DATA_PROCESSED / f"factor_kept{suffix}.parquet"
    removed_path = DATA_PROCESSED / f"factor_removed{suffix}.parquet"
    kept_df.to_parquet(kept_path, index=False)
    removed_df.to_parquet(removed_path, index=False)
    print(f"✅ 保留因子明细: {kept_path}")
    print(f"✅ 剔除因子明细: {removed_path}")

    # 3. 人类可读报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"factor_redundancy_report{suffix}.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("因子冗余剔除报告（日频 IC 相关性）")
    lines.append(f"阈值: 相关 > {threshold} 即剔除 | 因子总数: {total}")
    lines.append(f"|IC| ≥ {min_abs_ic} 过滤: 剔除 {n_filtered_ic} 个低信号因子")
    lines.append(f"ICIR > {min_icir} 过滤: 剔除 {n_filtered_icir} 个不稳定因子")
    lines.append("=" * 70)
    lines.append("")
    lines.extend(
        build_funnel(
            total,
            n_filtered_ic,
            n_filtered_icir,
            len(kept_df),
            len(removed_df),
            min_abs_ic,
            min_icir,
            threshold,
        )
    )
    lines.append("\n" + "-" * 70)
    lines.append(f"保留因子: {len(kept_df)} 个 ({(len(kept_df) / total) * 100:.1f}%)")
    lines.append(
        f"剔除因子: {len(removed_df)} 个（冗余剔除 {(len(removed_df) / (total - n_filtered)) * 100:.1f}%）"
    )
    lines.append("\n" + "-" * 70)
    lines.append("保留因子（按 IC 降序）:")
    lines.append("-" * 70)
    for _, r in kept_df.iterrows():
        lines.append(f"  {r['factor']:<24} IC={r['ic_mean']:+.4f}  ICIR={r['icir']:+.4f}")
    if not removed_df.empty:
        lines.append("\n" + "-" * 70)
        lines.append("剔除因子（冗余明细）:")
        lines.append("-" * 70)
        for _, r in removed_df.iterrows():
            lines.append(
                f"  {r['factor']:<24} IC={r['ic_mean']:+.4f}  "
                f"与 {r['redundant_with']} 相关 {r['corr']:.3f} "
                f"(该因子 IC={r['redundant_with_ic']:+.4f})"
            )
    lines.append("\n" + "=" * 70)
    lines.append(f"报告已生成: {report_path}")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"✅ 冗余报告: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="按标签剔除冗余因子")
    parser.add_argument("threshold", nargs="?", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("min_abs_ic", nargs="?", type=float, default=0.01)
    parser.add_argument("min_icir", nargs="?", type=float, default=0.05)
    parser.add_argument("--input-prefix", default="", help="读取 <prefix>_daily_rankic/evaluation")
    parser.add_argument("--output-suffix", default="", help="输出 selected_factor_cols_<suffix>.json")
    parser.add_argument("--label", default=None, help="写入产物元数据，防止跨标签误用")
    args = parser.parse_args()
    threshold = args.threshold
    min_abs_ic = args.min_abs_ic
    min_icir = args.min_icir

    daily_ic, summary = load_inputs(args.input_prefix)
    total = len(summary)

    # 1. |IC| + |ICIR| 过滤（低信号/不稳定因子不进模型）
    summary, n_filtered_ic, n_filtered_icir = filter_by_abs_ic(summary, min_abs_ic, min_icir)
    print(
        f"📊 因子总数: {total}，|IC| ≥ {min_abs_ic} 且 |ICIR| > {min_icir} 过滤后: {len(summary)} 个"
        f"（|IC| 过滤 {n_filtered_ic} 个，|ICIR| 过滤 {n_filtered_icir} 个）"
    )
    if len(summary) == 0:
        print("[ERROR] 过滤后无因子，请调低 min_abs_ic / min_icir 或检查评估结果")
        return 1

    # 2. 冗余剔除（在过滤后的因子上进行）
    print(f"📊 冗余剔除相关阈值: {threshold}")
    kept_df, removed_df = greedy_prune(daily_ic, summary, threshold)

    save_results(
        kept_df,
        removed_df,
        threshold,
        total,
        n_filtered_ic,
        n_filtered_icir,
        min_abs_ic,
        min_icir,
        args.output_suffix,
        args.label,
    )

    # 终端摘要
    print("\n" + "=" * 70)
    print(
        f"🏆 保留 {len(kept_df)} / {total} 个因子 "
        f"(过滤 {n_filtered_ic} 个低 |IC| + {n_filtered_icir} 个低 |ICIR|，冗余剔除 {len(removed_df)} 个)"
    )
    print("\n" + "\n".join(build_funnel(
        total,
        n_filtered_ic,
        n_filtered_icir,
        len(kept_df),
        len(removed_df),
        min_abs_ic,
        min_icir,
        threshold,
    )))
    if not removed_df.empty:
        print(f"\n🗑️ 剔除 Top 10（按相关度降序）:")
        top = removed_df.sort_values("corr", ascending=False).head(10)
        for _, r in top.iterrows():
            print(f"   {r['factor']:<24} 与 {r['redundant_with']:<24} 相关 {r['corr']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
