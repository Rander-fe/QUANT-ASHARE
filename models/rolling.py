"""Ridge/MLP 共用的严格滚动训练与评估工具。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import LGB_REPORTS_DIR, PREDICTIONS_DIR, TEST_PERIOD, VALID_PERIOD
from models.lightgbm.config import ROLLING
from models.lightgbm.data import build_rolling_windows, label_horizon, load_factor_list, prepare_data
from models.lightgbm.evaluate import calc_ic, evaluate_predictions, save_daily_ic


def segment_of(date: pd.Timestamp) -> str:
    if pd.Timestamp(VALID_PERIOD[0]) <= date <= pd.Timestamp(VALID_PERIOD[1]):
        return "valid"
    if pd.Timestamp(TEST_PERIOD[0]) <= date <= pd.Timestamp(TEST_PERIOD[1]):
        return "test"
    return "other"


def load_inputs(label: str) -> tuple[pd.DataFrame, list[str]]:
    factors = load_factor_list(label_col=label)
    if factors:
        print(f"[INFO] 使用精选因子: {len(factors)} 个")
    return prepare_data(label_col=label, factor_list=factors)


def rolling_slices(df: pd.DataFrame, label: str, step: int, train_len: int,
                   valid_len: int, quick: bool = False, max_steps: int | None = None):
    dates = np.sort(df["date"].unique())
    windows = build_rolling_windows(
        dates, {"step": step, "train_len": train_len, "valid_len": valid_len},
        purge_days=label_horizon(label),
    )
    if quick:
        windows = windows[-3:]
    if max_steps:
        windows = windows[-max_steps:]
    print(f"[INFO] 滚动窗口: {len(windows)}，边界隔离: {label_horizon(label)} 日")
    for i, (tr, va, te) in enumerate(windows):
        tr_dates, va_dates, te_dates = dates[tr], dates[va], dates[te]
        yield i, df[df["date"].isin(tr_dates)], df[df["date"].isin(va_dates)], \
            df[df["date"].isin(te_dates)], tr_dates, va_dates, te_dates


def valid_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    mask = frame[label].notna() & np.isfinite(frame[label])
    if {"limit_up", "limit_down"} <= set(frame.columns):
        mask &= ~frame[["limit_up", "limit_down"]].fillna(False).any(axis=1)
    return frame.loc[mask]


def sample_rows(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum and len(frame) > maximum:
        return frame.sample(n=maximum, random_state=seed)
    return frame


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """为不原生支持缺失值的模型生成有限值 float32 矩阵。"""
    values = frame[features].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def finish_run(predictions: list[pd.DataFrame], model_name: str, label: str,
               eval_segments: list[str], partial: bool = False) -> Path:
    pred = pd.concat(predictions, ignore_index=True)
    suffix = "_smoke" if partial else ""
    out = PREDICTIONS_DIR / f"{model_name}_pred_{label}{suffix}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pred[["symbol", "date", "pred", "segment"]].to_parquet(out, index=False)
    print(f"[OK] 预测保存: {out}")
    for seg in eval_segments:
        part = pred[pred["segment"] == seg]
        if part.empty:
            continue
        metrics = evaluate_predictions(part, "pred", label)
        print(f"[{model_name}/{seg}] RankIC={metrics['rank_ic_mean']} "
              f"IC={metrics['ic_mean']} RankICIR={metrics['rank_icir']}")
        daily = calc_ic(part, "pred", label)
        report_group = "smoke" if partial else label
        save_daily_ic(daily, LGB_REPORTS_DIR / model_name / report_group / f"daily_ic_{seg}.parquet")
    return out
