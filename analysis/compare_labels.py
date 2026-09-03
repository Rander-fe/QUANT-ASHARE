# -*- coding: utf-8 -*-
"""三标签（5/10/20 日）LightGBM 验证集对比分析。

用法（在三个标签各自跑完 train_lgb 后）：
    python analysis/compare_labels.py
    python analysis/compare_labels.py --seg valid

读取 data/processed/predictions/lgb_pred_{label}.parquet，
用验证集（铁律区间 2023-05~2025-01）评估 IC/ICIR/RankIC/RankICIR/分层收益，
输出对比表 -> reports/lightgbm/label_comparison.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_PROCESSED, PREDICTIONS_DIR, REPORTS_DIR
from models.lightgbm.evaluate import evaluate_predictions

LABELS = ["label_ret_5", "label_ret_10", "label_ret_20"]


def main() -> int:
    parser = argparse.ArgumentParser(description="标签预测评估（验证集对比 / 最终测试集评估）")
    parser.add_argument("--seg", default="valid", choices=("valid", "test"),
                        help="评估区段（默认 valid；test 仅供最终评估一次）")
    parser.add_argument("--label", default=None, choices=LABELS,
                        help="只评估指定标签（默认全部；test 段铁律：仅评估验证集选定的最优标签一次）")
    args = parser.parse_args()

    rows = []
    labels_path = DATA_PROCESSED / "labels.parquet"
    labels_df = pd.read_parquet(labels_path, columns=["symbol", "date", *LABELS])
    labels_df["date"] = pd.to_datetime(labels_df["date"])
    labels_to_eval = [args.label] if args.label else LABELS
    for label in labels_to_eval:
        path = PREDICTIONS_DIR / f"lgb_pred_{label}.parquet"
        if not path.exists():
            print(f"[SKIP] 缺少 {path}，跳过 {label}")
            continue
        df = pd.read_parquet(path)
        seg_df = df[df["segment"] == args.seg]
        if label not in seg_df.columns:
            seg_df = seg_df.merge(labels_df[["symbol", "date", label]],
                                  on=["symbol", "date"], how="left", validate="one_to_one")
        if seg_df.empty:
            print(f"[SKIP] {label} 无 {args.seg} 段预测（{path}）")
            continue
        m = evaluate_predictions(seg_df, "pred", label)
        print(f"[{label}] {args.seg} {seg_df['date'].min().date()}~{seg_df['date'].max().date()} "
              f"IC={m['ic_mean']} ICIR={m['icir']} RankIC={m['rank_ic_mean']} "
              f"重叠RankICIR={m['rank_icir_daily_overlapping']} "
              f"NW-t={m['rank_ic_nw_t']} 非重叠RankIC中位数={m['rank_ic_nonoverlap_median']} "
              f"TOP组={m['top_group_mean']} 多空={m['long_short_spread']}")
        rows.append({
            "label": label,
            "segment": args.seg,
            "n_days": m["days"],
            "ic_mean": m["ic_mean"],
            "icir": m["icir"],
            "rank_ic_mean": m["rank_ic_mean"],
            "rank_icir": m["rank_icir"],
            "rank_icir_daily_overlapping": m["rank_icir_daily_overlapping"],
            "rank_ic_nw_t": m["rank_ic_nw_t"],
            "rank_ic_nw_lags": m["rank_ic_nw_lags"],
            "rank_ic_nonoverlap_median": m["rank_ic_nonoverlap_median"],
            "rank_ic_nonoverlap_worst": m["rank_ic_nonoverlap_worst"],
            "rank_ic_nonoverlap_best": m["rank_ic_nonoverlap_best"],
            "rank_ic_nonoverlap_positive_ratio": m["rank_ic_nonoverlap_positive_ratio"],
            "rank_icir_nonoverlap_median": m["rank_icir_nonoverlap_median"],
            "nonoverlap_min_observations": m["nonoverlap_min_observations"],
            "nonoverlap_max_observations": m["nonoverlap_max_observations"],
            "ic_positive_ratio": m["ic_positive_ratio"],
            "top_group_mean": m["top_group_mean"],
            "bottom_group_mean": m["bottom_group_mean"],
            "long_short_spread": m["long_short_spread"],
        })

    if not rows:
        print("[WARN] 没有任何标签的预测可供对比，请先运行 train_lgb（三个标签各一次）")
        return 1

    out = pd.DataFrame(rows)
    # 按验证集 RankIC 排序（模型选择主指标）
    if args.seg == "valid":
        out = out.sort_values("rank_ic_mean", ascending=False)
    scope = args.label or "all"
    # 区段和评估范围进入文件名，避免单标签运行覆盖完整比较表。
    out_path = REPORTS_DIR / "lightgbm" / f"label_comparison_{args.seg}_{scope}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 对比表已保存: {out_path}")
    print("\n指标口径：重叠RankICIR仅作描述；NW-t修正标签重叠；非重叠指标汇总全部调仓起点。")
    print("选择建议：以验证集 RankIC 为主，并要求 NW-t、非重叠 RankIC 与组合回测共同支持；重叠 RankICIR 不单独用于选择。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
