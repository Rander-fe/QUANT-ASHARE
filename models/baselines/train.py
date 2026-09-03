"""Purged Walk-Forward 等权与训练期 IC 加权因子合成基准。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_PROCESSED, PREDICTIONS_DIR, REPORTS_DIR
from models.lightgbm.config import LABEL_COL, ROLLING
from models.rolling import finish_run, load_inputs, rolling_slices, segment_of, valid_rows
from research.protocol import write_experiment_manifest


def calculate_daily_ic(frame: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    """一次性计算日频 RankIC 矩阵，供所有滚动窗口复用。"""
    values = {}
    for feature in features:
        values[feature] = frame.groupby("date", sort=False).apply(
            lambda g: g[feature].corr(g[label], method="spearman"),
            include_groups=False,
        )
    daily = pd.DataFrame(values)
    daily.index = pd.to_datetime(daily.index)
    return daily.replace([np.inf, -np.inf], np.nan).sort_index()


def estimate_ic_weights(daily_ic: pd.DataFrame, train_dates: np.ndarray) -> pd.Series:
    """仅汇总当前训练窗口内的日频 RankIC，并做 L1 归一化。"""
    weights = daily_ic.reindex(pd.to_datetime(train_dates)).mean().fillna(0.0)
    scale = float(weights.abs().sum())
    return weights / scale if scale > 1e-12 else pd.Series(1.0 / len(features), index=features)


def score_frame(frame: pd.DataFrame, features: list[str], weights: pd.Series) -> np.ndarray:
    values = frame[features].to_numpy(dtype=np.float64, copy=True)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values @ weights.reindex(features).to_numpy(dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="等权与训练期IC加权因子基准")
    parser.add_argument("--label", default=LABEL_COL)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-segments", default="valid")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    segments = [x.strip() for x in args.eval_segments.split(",") if x.strip()]
    if "test" in segments and not args.confirm_final_test:
        parser.error("测试集评估需显式传入 --confirm-final-test")

    df, features = load_inputs(args.label)
    print("[INFO] 一次性计算日频 RankIC 矩阵")
    daily_ic = calculate_daily_ic(valid_rows(df, args.label), features, args.label)
    predictions = {"equal_weight": [], "ic_weight": []}
    for _, train, _, test, tr_dates, _, _ in rolling_slices(
        df, args.label, ROLLING["step"], ROLLING["train_len"], ROLLING["valid_len"],
        args.quick, args.max_steps,
    ):
        ic_weights = estimate_ic_weights(daily_ic, tr_dates)
        direction = np.sign(ic_weights).replace(0.0, np.nan)
        if direction.notna().any():
            equal = direction.fillna(0.0) / float(direction.notna().sum())
        else:
            equal = pd.Series(1.0 / len(features), index=features)
        for name, weights in (("equal_weight", equal), ("ic_weight", ic_weights)):
            frame = test[["symbol", "date", args.label]].copy()
            frame["pred"] = score_frame(test, features, weights)
            frame["segment"] = frame["date"].map(segment_of)
            predictions[name].append(frame)

    partial = args.quick or args.max_steps is not None
    for name, frames in predictions.items():
        finish_run(frames, name, args.label, segments, partial=partial)
        manifest = REPORTS_DIR / "model_comparison" / "manifests" / f"{name}_{args.label}.json"
        write_experiment_manifest(
            manifest, experiment=name,
            config={"label": args.label, "rolling": ROLLING, "segments": segments,
                    "partial": partial, "weight_source": "train_only",
                    "equal_weight_definition": "equal_after_train_ic_direction_alignment"},
            inputs=[DATA_PROCESSED / "features.parquet", DATA_PROCESSED / "labels.parquet"],
            features=features, test_data_used="test" in segments,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
