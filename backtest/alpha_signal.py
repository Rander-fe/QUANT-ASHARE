"""模型预测到可投资复合 Alpha 的横截面处理。"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SIGNAL_COLUMNS = {"symbol", "date", "pred", "industry", "circ_mv"}


def neutralize_model_score(frame: pd.DataFrame, winsor_quantile: float = 0.01) -> pd.DataFrame:
    """对单日模型分数做缩尾、行业/对数市值中性化和 Z-Score。

    输出端再次中性化是有意的：即使输入因子已经中性化，非线性模型仍可能
    重新形成行业和市值暴露。回归只使用同一日横截面信息。
    """
    missing = REQUIRED_SIGNAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"信号截面缺少字段: {sorted(missing)}")
    out = frame.copy()
    score = pd.to_numeric(out["pred"], errors="coerce")
    mv = pd.to_numeric(out["circ_mv"], errors="coerce")
    valid = score.notna() & np.isfinite(score) & mv.gt(0) & out["industry"].notna()
    out = out.loc[valid].copy()
    if len(out) < 20:
        return out.assign(alpha=np.nan).iloc[0:0]

    lo, hi = out["pred"].quantile([winsor_quantile, 1 - winsor_quantile])
    y = out["pred"].clip(lo, hi).astype(float).to_numpy()
    log_mv = np.log(out["circ_mv"].astype(float).to_numpy())
    industries = pd.get_dummies(out["industry"].astype(str), drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(out)), log_mv, industries.to_numpy()])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    std = residual.std(ddof=1)
    out["alpha"] = (residual - residual.mean()) / (std if std > 1e-12 else 1.0)
    out["rank"] = out["alpha"].rank(method="first", ascending=False).astype(int)
    return out.sort_values("rank")


def select_with_buffer(
    ranked: pd.DataFrame,
    current_symbols: set[str],
    topk: int,
    buffer: int,
    max_drop: int,
    max_industry_weight: float = 0.20,
) -> list[str]:
    """TopK + buffer + dropout，降低无意义的边界换手。"""
    if ranked.empty:
        return []
    ranked = ranked.sort_values("rank")
    rank_map = dict(zip(ranked["symbol"], ranked["rank"]))
    forced_out = {s for s in current_symbols if s not in rank_map}
    voluntary = sorted(
        (s for s in current_symbols if rank_map.get(s, topk + buffer + 1) > topk + buffer),
        key=lambda s: rank_map.get(s, 10**9), reverse=True,
    )[:max_drop]
    retained = sorted(
        (s for s in current_symbols if s not in forced_out and s not in voluntary),
        key=lambda s: (rank_map.get(s, 10**9), s),
    )
    selected = list(dict.fromkeys(retained))
    industry_of = dict(zip(ranked["symbol"], ranked["industry"].astype(str)))
    industry_cap = max(1, int(np.floor(topk * max_industry_weight)))
    industry_count: dict[str, int] = {}
    for symbol in selected:
        industry = industry_of.get(symbol, "UNKNOWN")
        industry_count[industry] = industry_count.get(industry, 0) + 1
    for symbol in ranked["symbol"]:
        if len(selected) >= topk:
            break
        if symbol in selected:
            continue
        industry = industry_of.get(symbol, "UNKNOWN")
        if industry_count.get(industry, 0) >= industry_cap:
            continue
        selected.append(symbol)
        industry_count[industry] = industry_count.get(industry, 0) + 1
    return selected[:topk]
