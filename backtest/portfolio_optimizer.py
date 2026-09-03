"""带风险、行业和换手约束的长仓组合优化。"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from backtest.config import PortfolioConfig


def estimate_covariance(
    close: pd.DataFrame,
    symbols: list[str],
    signal_date: pd.Timestamp,
    config: PortfolioConfig,
) -> np.ndarray:
    """只使用信号日及以前的收盘价估计年化收缩协方差。"""
    prices = close.loc[:signal_date, symbols].tail(config.covariance_lookback + 1)
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    # 缺失收益按当日横截面中位数填充，避免把停牌机械当成零波动。
    returns = returns.T.fillna(returns.median(axis=1)).T.fillna(0.0)
    if len(returns) < config.covariance_min_observations:
        variance = returns.var(ddof=1).reindex(symbols).fillna(0.02 ** 2).clip(1e-6)
        return np.diag(variance.to_numpy() * 252.0)
    covariance = LedoitWolf().fit(returns.to_numpy()).covariance_ * 252.0
    return (covariance + covariance.T) / 2.0


def optimize_weights(
    ranked: pd.DataFrame,
    selected: list[str],
    covariance: np.ndarray,
    current_weights: dict[str, float],
    exposure: float,
    config: PortfolioConfig,
) -> dict[str, float]:
    """最大化 Alpha - 风险 - 交易惩罚，并施加可解释的组合约束。"""
    frame = ranked.set_index("symbol").loc[selected]
    alpha = frame["alpha"].to_numpy(dtype=float)
    alpha = np.clip(alpha, -3.0, 3.0) * config.alpha_scale
    previous = np.array([current_weights.get(s, 0.0) for s in selected])
    outside_weight = sum(w for s, w in current_weights.items() if s not in set(selected))
    n = len(selected)
    w = cp.Variable(n)
    turnover = cp.norm1(w - previous) + outside_weight
    objective = cp.Maximize(
        alpha @ w
        - config.risk_aversion * cp.quad_form(w, cp.psd_wrap(covariance))
        - config.turnover_penalty * turnover
    )
    constraints = [w >= 0, w <= config.max_stock_weight, cp.sum(w) == exposure]
    industries = frame["industry"].astype(str).to_numpy()
    for industry in np.unique(industries):
        constraints.append(cp.sum(w[industries == industry]) <= config.max_industry_weight)
    if current_weights:
        constraints.append(turnover <= config.max_rebalance_turnover)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except cp.error.SolverError:
        problem.solve(solver="SCS", verbose=False)
    if w.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        # 约束因停牌/旧持仓而暂时不可行时，保留风险与行业约束，仅放宽换手硬上限。
        fallback = cp.Problem(objective, constraints[:-1] if current_weights else constraints)
        fallback.solve(solver="CLARABEL", verbose=False)
    if w.value is None:
        return {s: exposure / n for s in selected}
    values = np.maximum(np.asarray(w.value).reshape(-1), 0.0)
    values *= exposure / max(values.sum(), 1e-12)
    return dict(zip(selected, values.tolist()))
