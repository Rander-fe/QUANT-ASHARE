"""Purged Walk-Forward Ridge 线性回归基线。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MODELS_RIDGE_DIR
from models.lightgbm.config import LABEL_COL, ROLLING
from models.rolling import (feature_matrix, finish_run, load_inputs, rolling_slices,
                            sample_rows, segment_of, valid_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ridge 滚动选股基线")
    parser.add_argument("--label", default=LABEL_COL)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-train-samples", type=int, default=1_000_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-segments", default="valid")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    segments = [x.strip() for x in args.eval_segments.split(",") if x.strip()]
    if "test" in segments and not args.confirm_final_test:
        parser.error("测试集评估需显式传入 --confirm-final-test")

    df, features = load_inputs(args.label)
    partial = args.quick or args.max_steps is not None
    predictions = []
    for i, train, _, test, tr_dates, _, te_dates in rolling_slices(
        df, args.label, ROLLING["step"], ROLLING["train_len"], ROLLING["valid_len"],
        args.quick, args.max_steps,
    ):
        train = sample_rows(valid_rows(train, args.label), args.max_train_samples, 42 + i)
        model = Ridge(alpha=args.alpha, solver="lsqr")
        model.fit(feature_matrix(train, features), train[args.label].to_numpy(np.float32))
        pred = model.predict(feature_matrix(test, features))
        frame = test[["symbol", "date", args.label]].copy()
        frame["pred"] = pred
        frame["segment"] = frame["date"].map(segment_of)
        predictions.append(frame)
        model_dir = MODELS_RIDGE_DIR / ("smoke" if partial else args.label) / f"step_{i:03d}"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": features}, model_dir / "model.joblib")
        if (i + 1) % 5 == 0:
            print(f"[INFO] Ridge {i + 1}: train {tr_dates[0]}~{tr_dates[-1]}, test {te_dates[0]}~{te_dates[-1]}")
    finish_run(predictions, "ridge", args.label, segments, partial=partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
