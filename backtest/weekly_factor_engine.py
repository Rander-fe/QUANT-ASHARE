"""Reusable weekly factor portfolio engine for A-share candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WeeklyResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    schedule: pd.DataFrame
    metrics: dict


def weekly_schedule(trading_dates, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Last actual trading day in each ISO week, executed next trading day."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(trading_dates).dropna().unique())))
    dates = dates[(dates >= start) & (dates <= end)]
    calendar = pd.DataFrame({"date": dates})
    iso = calendar["date"].dt.isocalendar()
    signal_dates = calendar.groupby([iso.year, iso.week], sort=True)["date"].max().tolist()
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    rows = [{"signal_date": d, "execution_date": next_date[d]}
            for d in signal_dates if d in next_date and next_date[d] <= end]
    return pd.DataFrame(rows)


def _can_trade(row: pd.Series, side: str) -> bool:
    price, volume = row.get("open"), row.get("volume")
    if pd.isna(price) or not np.isfinite(float(price)) or float(price) <= 0:
        return False
    if pd.isna(volume) or not np.isfinite(float(volume)) or float(volume) <= 0:
        return False
    if side == "buy":
        return not bool(row.get("open_at_limit_up", False))
    return not bool(row.get("open_at_limit_down", False))


def _metrics(nav: pd.DataFrame, schedule: pd.DataFrame, trades: pd.DataFrame) -> dict:
    daily = nav["nav"].pct_change().dropna()
    benchmark = nav["benchmark_nav"].pct_change().dropna()
    years = max(len(daily) / 252, 1 / 252)
    annual = (nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (1 / years) - 1
    bench_annual = (nav["benchmark_nav"].iloc[-1] / nav["benchmark_nav"].iloc[0]) ** (1 / years) - 1
    drawdown = nav["nav"] / nav["nav"].cummax() - 1
    rebalance = nav.loc[nav["is_execution"], "one_way_turnover"].iloc[1:]
    blocked = schedule[["blocked_buy_count", "blocked_sell_count", "selected_count"]].sum()
    return {
        "annual_return": float(annual), "benchmark_annual_return": float(bench_annual),
        "annual_excess_return": float(annual - bench_annual),
        "annual_volatility": float(daily.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(daily.mean() / (daily.std(ddof=1) + 1e-12) * np.sqrt(252)),
        "information_ratio": float((daily - benchmark).mean() / ((daily - benchmark).std(ddof=1) + 1e-12) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()), "final_nav": float(nav["nav"].iloc[-1]),
        "rebalance_count": int(len(schedule)), "average_weekly_one_way_turnover": float(rebalance.mean()),
        "annualized_one_way_turnover": float(rebalance.sum() / years),
        "total_trading_cost": float(trades["cost"].sum()) if not trades.empty else 0.0,
        "blocked_buy_count": int(blocked["blocked_buy_count"]),
        "blocked_sell_count": int(blocked["blocked_sell_count"]),
        "blocked_buy_rate_per_selection": float(blocked["blocked_buy_count"] / max(blocked["selected_count"], 1)),
    }


def run_weekly_factor_backtest(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    score_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: int,
    long_quantile: float = .10,
    initial_cash: float = 100_000_000,
    lot_size: int = 100,
    minimum_cross_section: int = 100,
) -> WeeklyResult:
    signals, market, benchmark = signals.copy(), market.copy(), benchmark.copy()
    for frame in (signals, market, benchmark):
        frame["date"] = pd.to_datetime(frame["date"])
    if max(signals["date"].max(), market["date"].max()) > end:
        raise ValueError("Input exceeds frozen research-period end")
    dates = sorted(set(market["date"]))
    schedule = weekly_schedule(dates, start, end)
    execution_to_signal = dict(zip(schedule["execution_date"], schedule["signal_date"]))
    signal_by_date = {d: x for d, x in signals.groupby("date", observed=True)}
    market_by_date = {d: x.set_index("symbol", drop=False) for d, x in market.groupby("date", observed=True)}
    bench = benchmark.set_index("date")["close"].sort_index().reindex(dates).ffill()
    bench_nav = bench / bench.dropna().iloc[0]
    fee_rate = cost_bps / 10_000
    cash, positions, last_close = float(initial_cash), {}, {}
    nav_rows, trade_rows, holding_rows, schedule_rows = [], [], [], []

    for date in dates:
        if date < start or date > end:
            continue
        day = market_by_date[date]

        def price(symbol, field):
            if symbol in day.index and pd.notna(day.loc[symbol].get(field)):
                value = float(day.loc[symbol][field])
                if value > 0:
                    return value
            return last_close.get(symbol, 0.0)

        equity_open = cash + sum(q * price(s, "open") for s, q in positions.items())
        traded, day_cost = 0.0, 0.0
        is_execution = date in execution_to_signal
        if is_execution:
            signal_date = execution_to_signal[date]
            snapshot = signal_by_date.get(signal_date, pd.DataFrame())
            snapshot = snapshot.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
            if len(snapshot) >= minimum_cross_section:
                count = max(1, int(np.floor(len(snapshot) * long_quantile)))
                selected = snapshot.head(count)["symbol"].tolist()
            else:
                selected = []
            target_weight = 1 / len(selected) if selected else 0.0
            blocked_buy = blocked_sell = 0
            # Sell first. Untradable holdings remain in the portfolio.
            for symbol in list(positions):
                target_value = equity_open * target_weight if symbol in selected else 0.0
                open_price = price(symbol, "open")
                desired = int(target_value / open_price // lot_size * lot_size) if open_price > 0 else positions[symbol]
                sell_qty = max(0, positions[symbol] - desired)
                if sell_qty <= 0:
                    continue
                if symbol not in day.index or not _can_trade(day.loc[symbol], "sell"):
                    blocked_sell += 1
                    continue
                notional, cost = sell_qty * open_price, sell_qty * open_price * fee_rate
                cash += notional - cost
                positions[symbol] -= sell_qty
                if positions[symbol] == 0:
                    positions.pop(symbol)
                traded += notional
                day_cost += cost
                trade_rows.append({"date": date, "signal_date": signal_date, "symbol": symbol,
                                   "side": "sell", "shares": sell_qty, "price": open_price,
                                   "notional": notional, "cost": cost, "cost_bps": cost_bps})
            # Buy toward equal weight using remaining cash.
            for symbol in selected:
                if symbol not in day.index or not _can_trade(day.loc[symbol], "buy"):
                    blocked_buy += 1
                    continue
                open_price = price(symbol, "open")
                desired = int((equity_open * target_weight) / open_price // lot_size * lot_size)
                buy_qty = max(0, desired - positions.get(symbol, 0))
                affordable = int((cash / (open_price * (1 + fee_rate))) // lot_size * lot_size)
                buy_qty = min(buy_qty, affordable)
                if buy_qty <= 0:
                    continue
                notional, cost = buy_qty * open_price, buy_qty * open_price * fee_rate
                cash -= notional + cost
                positions[symbol] = positions.get(symbol, 0) + buy_qty
                traded += notional
                day_cost += cost
                trade_rows.append({"date": date, "signal_date": signal_date, "symbol": symbol,
                                   "side": "buy", "shares": buy_qty, "price": open_price,
                                   "notional": notional, "cost": cost, "cost_bps": cost_bps})
            schedule_rows.append({"signal_date": signal_date, "execution_date": date,
                                  "selected_count": len(selected), "actual_holdings": len(positions),
                                  "blocked_buy_count": blocked_buy, "blocked_sell_count": blocked_sell,
                                  "two_sided_turnover": traded / max(equity_open, 1e-12),
                                  "one_way_turnover": traded / max(2 * equity_open, 1e-12)})

        equity_close = cash + sum(q * price(s, "close") for s, q in positions.items())
        # Only held symbols need a stale-price cache. Iterating the full market
        # every day makes multi-factor research unnecessarily expensive.
        for symbol in positions:
            if symbol in day.index and pd.notna(day.loc[symbol].get("close")):
                close_value = float(day.loc[symbol]["close"])
                if close_value > 0:
                    last_close[symbol] = close_value
        one_way = traded / max(2 * equity_open, 1e-12)
        nav_rows.append({"date": date, "nav": equity_close / initial_cash,
                         "benchmark_nav": float(bench_nav.loc[date]), "equity": equity_close,
                         "cash": cash, "n_holdings": len(positions), "is_execution": is_execution,
                         "one_way_turnover": one_way, "cost": day_cost})
        # Weekly snapshots are sufficient to reconstruct target holdings and
        # keep artifacts compact; NAV remains daily.
        if is_execution:
            for symbol, qty in positions.items():
                holding_rows.append({"date": date, "symbol": symbol, "shares": qty,
                                     "close": price(symbol, "close"),
                                     "market_value": qty * price(symbol, "close"), "cost_bps": cost_bps})
    nav, trades, holdings, schedule_out = map(pd.DataFrame, (nav_rows, trade_rows, holding_rows, schedule_rows))
    return WeeklyResult(nav, trades, holdings, schedule_out, _metrics(nav, schedule_out, trades))
