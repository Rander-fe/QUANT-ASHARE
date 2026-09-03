import numpy as np
import pandas as pd

from models.lightgbm.optuna_search_v2 import (
    ECONOMIC_WEIGHT,
    STABILITY_PENALTY,
    TOP_FRACTION,
    TRANSACTION_COST_BPS,
    _production_params,
    portfolio_utility,
)


def _portfolio_frame() -> pd.DataFrame:
    rows = []
    for day in range(4):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        for stock in range(20):
            # 预测排序正确且持仓稳定；Top 10% 是 S18/S19。
            rows.append({
                "date": date,
                "symbol": f"S{stock:02d}",
                "pred": float(stock),
                "label_ret_2": float(stock) / 100.0,
            })
    return pd.DataFrame(rows)


def test_portfolio_utility_uses_nonoverlap_and_turnover_cost() -> None:
    frame = _portfolio_frame()
    free = portfolio_utility(frame, "label_ret_2", horizon=2,
                             top_fraction=0.10, cost_bps=0.0)
    costly = portfolio_utility(frame, "label_ret_2", horizon=2,
                               top_fraction=0.10, cost_bps=100.0)

    assert free["rebalance_count"] == 2
    assert free["gross_excess_return"] > 0
    # 首次建仓=1，第二次持仓不变=0。
    assert np.isclose(free["turnover"], 0.5)
    assert costly["net_excess_utility"] < free["net_excess_utility"]


def test_production_params_remove_lightgbm_alias_conflicts() -> None:
    params = _production_params({
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
    })

    assert params["colsample_bytree"] == 0.7
    assert params["subsample"] == 0.8
    assert params["subsample_freq"] == 3
    assert "feature_fraction" not in params
    assert "bagging_fraction" not in params
    assert "bagging_freq" not in params


def test_scoring_policy_is_precommitted() -> None:
    assert STABILITY_PENALTY == 0.50
    assert ECONOMIC_WEIGHT == 0.05
    assert TRANSACTION_COST_BPS == 10.0
    assert TOP_FRACTION == 0.10
