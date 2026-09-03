"""基于日线的 A 股可交易组合回测引擎。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.config import PortfolioConfig
from backtest.alpha_signal import neutralize_model_score, select_with_buffer
from backtest.portfolio_optimizer import estimate_covariance, optimize_weights


@dataclass
class BacktestResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    metrics: dict


def _tradable(row: pd.Series, side: str) -> bool:
    price = row.get("open")
    volume = row.get("volume", 0)
    if (pd.isna(price) or float(price) <= 0 or pd.isna(volume)
            or not np.isfinite(float(volume)) or float(volume) <= 0):
        return False
    if side == "buy":
        return not bool(row.get("open_at_limit_up", False))
    return not bool(row.get("open_at_limit_down", False))


def _commission(notional: float, rate: float, minimum: float) -> float:
    return max(minimum, notional * rate) if notional > 0 else 0.0


def _risk_exposure(nav_rows: list[dict], equity_open: float, peak_equity: float,
                   config: PortfolioConfig) -> tuple[float, float, float]:
    """仅用此前净值计算波动率目标与回撤分级仓位。"""
    drawdown = equity_open / max(peak_equity, 1e-12) - 1.0
    if not config.use_dynamic_exposure:
        return 1.0, np.nan, drawdown
    realized_vol = np.nan
    vol_scale = 1.0
    if len(nav_rows) > config.volatility_lookback:
        values = pd.Series([row["nav"] for row in nav_rows[-config.volatility_lookback - 1:]])
        realized_vol = float(values.pct_change().dropna().std(ddof=1) * np.sqrt(252))
        if np.isfinite(realized_vol) and realized_vol > 1e-12:
            vol_scale = min(1.0, config.target_annual_volatility / realized_vol)
    drawdown_scale = 1.0
    for threshold, scale in config.drawdown_tiers:
        if drawdown <= threshold:
            drawdown_scale = min(drawdown_scale, scale)
    exposure = max(config.minimum_exposure, min(vol_scale, drawdown_scale, 1.0))
    return exposure, realized_vol, drawdown


def performance_metrics(nav: pd.DataFrame) -> dict:
    if nav.empty:
        return {}
    daily = nav["nav"].pct_change().dropna()
    bench = nav["benchmark_nav"].pct_change().dropna()
    years = max(len(daily) / 252.0, 1 / 252.0)
    annual_return = (nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (1 / years) - 1
    benchmark_return = (nav["benchmark_nav"].iloc[-1] / nav["benchmark_nav"].iloc[0]) ** (1 / years) - 1
    drawdown = nav["nav"] / nav["nav"].cummax() - 1
    excess = daily.align(bench, join="inner")[0] - daily.align(bench, join="inner")[1]
    cost_ratio = (nav["cost"] / nav["equity"].shift(1).fillna(nav["equity"])).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    if "is_initial_build" in nav.columns:
        rebalance_turnover = nav.loc[~nav["is_initial_build"], "turnover"]
    else:
        # 兼容旧产物：首次非零换手视为初始建仓。
        first_trade = nav.index[nav["turnover"] > 0]
        rebalance_turnover = nav["turnover"].copy()
        if len(first_trade):
            rebalance_turnover = rebalance_turnover.drop(first_trade[0])
    return {
        "annual_return": float(annual_return),
        "benchmark_annual_return": float(benchmark_return),
        "annual_excess_return": float(annual_return - benchmark_return),
        "annual_volatility": float(daily.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(daily.mean() / (daily.std(ddof=1) + 1e-12) * np.sqrt(252)),
        "information_ratio": float(excess.mean() / (excess.std(ddof=1) + 1e-12) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "average_daily_turnover": float(rebalance_turnover.sum() / max(len(nav), 1)),
        "annualized_two_sided_turnover": float(rebalance_turnover.sum() / years),
        "initial_build_turnover": float(nav.loc[nav.get("is_initial_build", False), "turnover"].sum()
                                        if "is_initial_build" in nav.columns else
                                        nav.loc[first_trade[0], "turnover"] if len(first_trade) else 0.0),
        "average_exposure": float(nav["target_exposure"].mean()),
        "average_actual_exposure": float(nav["actual_exposure"].mean())
        if "actual_exposure" in nav.columns else np.nan,
        "max_rebalance_turnover": float(rebalance_turnover.max()) if len(rebalance_turnover) else 0.0,
        "max_regular_rebalance_turnover": float(
            nav.loc[(~nav["is_initial_build"]) & (~nav["risk_reduction"]), "turnover"].max()
        ) if {"is_initial_build", "risk_reduction"} <= set(nav.columns) else np.nan,
        "total_cost": float(nav["cost"].sum()),
        "total_cost_ratio": float(cost_ratio.sum()),
        "annualized_cost_ratio": float(cost_ratio.sum() / years),
        "trading_days": int(len(nav)),
    }


def run_portfolio_backtest(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: PortfolioConfig,
) -> BacktestResult:
    """执行 T 日收盘出信号、下一交易日开盘成交的组合回测。"""
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    benchmark = benchmark.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])

    prediction_dates = pd.DatetimeIndex(predictions["date"].dropna().unique())
    if prediction_dates.empty:
        raise ValueError("预测数据为空")
    dates = pd.DatetimeIndex(sorted(
        d for d in set(market["date"])
        if prediction_dates.min() <= d <= prediction_dates.max()
    ))
    if config.rebalance_days <= 0:
        raise ValueError("rebalance_days 必须为正整数")
    if not 0 <= config.rebalance_offset < config.rebalance_days:
        raise ValueError("rebalance_offset 必须满足 0 <= offset < rebalance_days")
    if config.rebalance_rule == "calendar_week_end":
        calendar = pd.DataFrame({"date": pd.to_datetime(dates)})
        iso = calendar["date"].dt.isocalendar()
        signal_dates = pd.DatetimeIndex(
            calendar.groupby([iso.year, iso.week], sort=True)["date"].max()
        )
    elif config.rebalance_rule == "every_n_trading_days":
        signal_dates = dates[config.rebalance_offset:: config.rebalance_days]
    else:
        raise ValueError(f"未知调仓规则: {config.rebalance_rule}")
    next_date = {pd.Timestamp(dates[i]): pd.Timestamp(dates[i + 1]) for i in range(len(dates) - 1)}
    available_predictions = set(prediction_dates)
    execution_signals = {next_date[pd.Timestamp(d)]: pd.Timestamp(d) for d in signal_dates
                         if pd.Timestamp(d) in next_date and pd.Timestamp(d) in available_predictions}

    market_by_date = {d: g.set_index("symbol", drop=False) for d, g in market.groupby("date")}
    close_history = market.pivot(index="date", columns="symbol", values="close").sort_index()
    pred_by_date = {d: g for d, g in predictions.groupby("date")}
    benchmark_close = benchmark.set_index("date")["close"].sort_index().reindex(dates).ffill()
    benchmark_nav = benchmark_close / benchmark_close.dropna().iloc[0]

    cash = config.initial_cash
    positions: dict[str, int] = {}
    last_price: dict[str, float] = {}
    nav_rows, trade_rows, holding_rows = [], [], []
    peak_equity = config.initial_cash
    previous_exposure = 1.0
    has_ever_built = False

    for date in dates:
        day = market_by_date.get(pd.Timestamp(date))
        if day is None:
            continue
        def price_of(symbol: str, field: str) -> float:
            if symbol in day.index and pd.notna(day.loc[symbol].get(field)):
                return float(day.loc[symbol][field])
            return last_price.get(symbol, 0.0)

        equity_open = cash + sum(qty * price_of(s, "open") for s, qty in positions.items())
        target_exposure, realized_vol, pretrade_drawdown = _risk_exposure(
            nav_rows, equity_open, peak_equity, config
        )
        traded_notional = 0.0
        day_cost = 0.0
        is_rebalance = pd.Timestamp(date) in execution_signals
        # 全程只能有一次免换手预算的初始建仓。组合中途清空后重新入场
        # 仍须遵守换手预算，否则可以通过“清仓 -> 重建”绕过年度约束。
        is_initial_build = is_rebalance and not has_ever_built
        risk_reduction = False

        if is_rebalance:
            signal_date = pd.Timestamp(execution_signals[pd.Timestamp(date)])
            snapshot = pred_by_date[signal_date]
            ranked = neutralize_model_score(snapshot, config.winsor_quantile)
            selected = select_with_buffer(
                ranked, set(positions), config.topk, config.buffer, config.max_drop,
                config.max_industry_weight,
            )
            current_weights = {
                symbol: qty * price_of(symbol, "open") / max(equity_open, 1e-12)
                for symbol, qty in positions.items()
            }
            if config.use_mean_variance_optimizer:
                covariance = estimate_covariance(
                    close_history, selected, signal_date, config
                )
                target_weights = optimize_weights(
                    ranked, selected, covariance, current_weights, target_exposure, config
                )
            else:
                # 等权是高噪声个股 Alpha 下更稳健的默认方案；仍保留行业、缓冲区和换手约束。
                target_weights = {s: target_exposure / len(selected) for s in selected}
            # 初次建仓不受换手预算限制；后续按双边成交额约束。
            turnover_budget = (float("inf") if is_initial_build
                               else equity_open * config.max_rebalance_turnover)
            risk_reduction = target_exposure < previous_exposure - 1e-12
            # 正常换仓必须为买单预留额度。旧实现允许卖单耗尽全部双边预算，
            # 导致无法买回、现金长期堆积。若实际卖出不足半数，剩余额度可给买单。
            sell_budget = (float("inf") if not np.isfinite(turnover_budget)
                           else turnover_budget / 2.0)
            sold_notional = 0.0
            bought_notional = 0.0

            # 先卖出或减仓，卖不出的股票继续持有并按收盘市值计价。
            for symbol in list(positions):
                target_value = equity_open * target_weights.get(symbol, 0.0)
                desired = int(target_value / max(price_of(symbol, "open"), 1e-12) // config.lot_size * config.lot_size) if symbol in selected else 0
                qty = positions[symbol]
                if desired >= qty or symbol not in day.index or not _tradable(day.loc[symbol], "sell"):
                    continue
                sell_qty = qty - desired
                if not risk_reduction and np.isfinite(turnover_budget):
                    budget_qty = int(max(0.0, sell_budget - sold_notional)
                                     / max(price_of(symbol, "open"), 1e-12)
                                     // config.lot_size * config.lot_size)
                    sell_qty = min(sell_qty, budget_qty)
                if sell_qty <= 0:
                    continue
                notional = sell_qty * price_of(symbol, "open")
                fee = _commission(notional, config.sell_cost, config.min_commission)
                cash += notional - fee
                # 预算可能只允许部分卖出；库存必须按实际成交量递减。
                # 旧实现直接写入 desired，会让未成交股份从资产账上凭空消失。
                remaining = qty - sell_qty
                positions[symbol] = remaining
                if remaining == 0:
                    positions.pop(symbol)
                traded_notional += notional
                sold_notional += notional
                day_cost += fee
                trade_rows.append({"date": date, "signal_date": signal_date, "symbol": symbol,
                                   "side": "sell", "shares": sell_qty, "price": price_of(symbol, "open"),
                                   "notional": notional, "cost": fee})

            # 再按优化权重由高到低买入；资金不足时自然缩减最后的订单。
            for symbol in sorted(selected, key=lambda s: target_weights.get(s, 0.0), reverse=True):
                if symbol not in day.index or not _tradable(day.loc[symbol], "buy"):
                    continue
                price = price_of(symbol, "open")
                current = positions.get(symbol, 0)
                target_value = equity_open * target_weights.get(symbol, 0.0)
                desired = int(target_value / price // config.lot_size * config.lot_size)
                buy_qty = max(0, desired - current)
                if np.isfinite(turnover_budget):
                    buy_budget = max(0.0, turnover_budget - sold_notional)
                    budget_qty = int(max(0.0, buy_budget - bought_notional)
                                     / price // config.lot_size * config.lot_size)
                    buy_qty = min(buy_qty, budget_qty)
                affordable = int((cash / (price * (1 + config.buy_cost))) // config.lot_size * config.lot_size)
                buy_qty = min(buy_qty, affordable)
                if buy_qty <= 0:
                    continue
                notional = buy_qty * price
                fee = _commission(notional, config.buy_cost, config.min_commission)
                cash -= notional + fee
                positions[symbol] = current + buy_qty
                traded_notional += notional
                bought_notional += notional
                day_cost += fee
                trade_rows.append({"date": date, "signal_date": signal_date, "symbol": symbol,
                                   "side": "buy", "shares": buy_qty, "price": price,
                                   "notional": notional, "cost": fee})

            if is_initial_build and positions:
                has_ever_built = True

        equity_close = cash + sum(qty * price_of(s, "close") for s, qty in positions.items())
        invested_close = sum(qty * price_of(s, "close") for s, qty in positions.items())
        peak_equity = max(peak_equity, equity_close)
        previous_exposure = target_exposure
        # 收盘估值完成后，才把当日收盘写入下一交易日的停牌估值缓存。
        for symbol, row in day.iterrows():
            if pd.notna(row.get("close")) and float(row["close"]) > 0:
                last_price[symbol] = float(row["close"])
        nav_rows.append({"date": date, "equity": equity_close,
                         "nav": equity_close / config.initial_cash,
                         "benchmark_nav": float(benchmark_nav.loc[date]),
                         "turnover": traded_notional / max(equity_open, 1e-12),
                         "is_rebalance": is_rebalance,
                         "is_initial_build": is_initial_build,
                         "risk_reduction": risk_reduction,
                         "cost": day_cost, "n_holdings": len(positions),
                         "target_exposure": target_exposure,
                         "actual_exposure": invested_close / max(equity_close, 1e-12),
                         "realized_volatility": realized_vol,
                         "pretrade_drawdown": pretrade_drawdown})
        for symbol, qty in positions.items():
            holding_rows.append({"date": date, "symbol": symbol, "shares": qty,
                                 "close": price_of(symbol, "close"),
                                 "market_value": qty * price_of(symbol, "close")})

    nav = pd.DataFrame(nav_rows)
    return BacktestResult(
        nav=nav,
        trades=pd.DataFrame(trade_rows),
        holdings=pd.DataFrame(holding_rows),
        metrics=performance_metrics(nav),
    )
