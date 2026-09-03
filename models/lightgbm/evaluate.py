# -*- coding: utf-8 -*-
"""预测评估：IC / ICIR / RankIC / RankICIR / 分层收益。

借鉴 Qlib 的 SigAnaRecord 指标口径：
    - IC     : 每日横截面 Pearson 相关系数
    - ICIR   : IC 均值 / IC 标准差（按 sqrt(252) 年化；多日标签下仅作描述）
    - RankIC : 每日横截面 Spearman 秩相关
    - RankICIR: RankIC 均值 / RankIC 标准差（年化；另报 HAC 与非重叠口径）
    - 分层   : 按预测值每日 10 分组，取 TOP 组平均收益（可选）
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def calc_ic(df: pd.DataFrame, pred_col: str, label_col: str) -> pd.DataFrame:
    """逐日计算横截面 IC / RankIC。

    Parameters
    ----------
    df : 长表 [symbol, date, pred_col, label_col]
    pred_col, label_col : 预测列 / 真实标签列

    Returns
    -------
    daily : DataFrame，索引为 date，含 ic / rank_ic 两列
    """
    rows = []
    for date, g in df.groupby("date"):
        if len(g) < 5:
            continue
        pred = g[pred_col].astype(float)
        label = g[label_col].astype(float)
        ic = pred.corr(label, method="pearson")
        rank_ic = pred.corr(label, method="spearman")
        rows.append((date, ic, rank_ic))
    daily = pd.DataFrame(rows, columns=["date", "ic", "rank_ic"]).set_index("date")
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    return daily


def _label_horizon(label_col: str) -> int:
    """从 ``label_ret_N`` 提取预测期，无法识别时退回 1 日。"""
    match = re.fullmatch(r"label_ret_(\d+)", label_col)
    return max(int(match.group(1)), 1) if match else 1


def _newey_west_mean_test(series: pd.Series, max_lags: int) -> tuple[float, float]:
    """返回样本均值的 Newey-West(HAC) 标准误与 t 值。

    使用 Bartlett 权重；该检验修正重叠前瞻收益造成的 IC 序列自相关。
    """
    x = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    max_lags = min(max(int(max_lags), 0), n - 2)
    demeaned = x - x.mean()
    long_run_var = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma
    se_mean = float(np.sqrt(max(long_run_var, 0.0) / n))
    t_stat = float(x.mean() / (se_mean + 1e-12))
    return se_mean, t_stat


def _nonoverlap_summary(series: pd.Series, horizon: int, ann_scaler: int) -> dict:
    """汇总所有调仓起点的非重叠 IC，避免单一起点依赖。"""
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    rows = []
    for offset in range(horizon):
        sample = clean.iloc[offset::horizon].dropna()
        if len(sample) < 2:
            continue
        mean = float(sample.mean())
        std = float(sample.std(ddof=1))
        rows.append({
            "mean": mean,
            "icir": mean / (std + 1e-12) * np.sqrt(ann_scaler / horizon),
            "n": int(len(sample)),
        })
    if not rows:
        return {
            "rank_ic_nonoverlap_median": np.nan,
            "rank_ic_nonoverlap_worst": np.nan,
            "rank_ic_nonoverlap_best": np.nan,
            "rank_ic_nonoverlap_positive_ratio": np.nan,
            "rank_icir_nonoverlap_median": np.nan,
            "nonoverlap_offsets": 0,
            "nonoverlap_min_observations": 0,
            "nonoverlap_max_observations": 0,
        }
    frame = pd.DataFrame(rows)
    return {
        "rank_ic_nonoverlap_median": round(float(frame["mean"].median()), 6),
        "rank_ic_nonoverlap_worst": round(float(frame["mean"].min()), 6),
        "rank_ic_nonoverlap_best": round(float(frame["mean"].max()), 6),
        "rank_ic_nonoverlap_positive_ratio": round(float((frame["mean"] > 0).mean()), 4),
        "rank_icir_nonoverlap_median": round(float(frame["icir"].median()), 6),
        "nonoverlap_offsets": int(len(frame)),
        "nonoverlap_min_observations": int(frame["n"].min()),
        "nonoverlap_max_observations": int(frame["n"].max()),
    }


def summarize_ic(daily: pd.DataFrame, ann_scaler: int = 252,
                 horizon: int = 1) -> dict:
    """汇总 IC，并对多日前瞻标签补充重叠修正口径。"""
    if daily.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "rank_ic_mean": np.nan, "rank_ic_std": np.nan, "rank_icir": np.nan,
                "ic_positive_ratio": np.nan, "days": 0,
                "label_horizon": int(horizon)}

    ic = daily["ic"]
    ric = daily["rank_ic"]
    nw_se, nw_t = _newey_west_mean_test(ric, max_lags=max(horizon - 1, 0))
    result = {
        "ic_mean": round(float(ic.mean()), 6),
        "ic_std": round(float(ic.std(ddof=1)), 6),
        "icir": round(float(ic.mean() / (ic.std(ddof=1) + 1e-12) * np.sqrt(ann_scaler)), 6),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_std": round(float(ric.std(ddof=1)), 6),
        "rank_icir": round(float(ric.mean() / (ric.std(ddof=1) + 1e-12) * np.sqrt(ann_scaler)), 6),
        "rank_icir_daily_overlapping": round(
            float(ric.mean() / (ric.std(ddof=1) + 1e-12) * np.sqrt(ann_scaler)), 6
        ),
        "rank_ic_nw_se": round(nw_se, 6),
        "rank_ic_nw_t": round(nw_t, 6),
        "rank_ic_nw_lags": int(max(horizon - 1, 0)),
        "label_horizon": int(horizon),
        "ic_positive_ratio": round(float((ic > 0).mean()), 4),
        "days": int(len(daily)),
    }
    result.update(_nonoverlap_summary(ric, horizon=horizon, ann_scaler=ann_scaler))
    return result


def top_group_return(df: pd.DataFrame, pred_col: str, label_col: str, n_groups: int = 10) -> dict:
    """每日按预测值分 n 组，返回 TOP 组平均收益（可作单调性参考）。

    注意：先剔除 label 缺失样本（标签是未来收益，数据末尾无未来值），
    否则尾部日期 TOP/BOTTOM 组均值会全为 NaN，污染整体统计。
    """
    top_rets, bottom_rets, spreads = [], [], []
    for _, g in df.groupby("date"):
        g = g[g[label_col].notna() & np.isfinite(g[label_col])]
        if len(g) < n_groups * 2:
            continue
        try:
            g = g.copy()
            g["grp"] = pd.qcut(g[pred_col].rank(method="first"), n_groups, labels=False)
        except ValueError:
            continue
        top_rets.append(g.loc[g["grp"] == n_groups - 1, label_col].mean())
        bottom_rets.append(g.loc[g["grp"] == 0, label_col].mean())
        spreads.append(top_rets[-1] - bottom_rets[-1])

    if not top_rets:
        return {"top_group_mean": np.nan, "bottom_group_mean": np.nan,
                "long_short_spread": np.nan, "groups": n_groups}
    return {
        "top_group_mean": round(float(np.mean(top_rets)), 6),
        "bottom_group_mean": round(float(np.mean(bottom_rets)), 6),
        "long_short_spread": round(float(np.mean(spreads)), 6),
        "groups": n_groups,
    }


def evaluate_predictions(df: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    """一键评估：IC 汇总 + 分层收益。"""
    daily = calc_ic(df, pred_col, label_col)
    summary = summarize_ic(daily, horizon=_label_horizon(label_col))
    summary.update(top_group_return(df, pred_col, label_col))
    return summary


def save_daily_ic(daily: pd.DataFrame, path: Path) -> None:
    """保存逐日 IC 序列。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_parquet(path, index=False)
