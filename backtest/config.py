"""组合回测的冻结参数。

研究阶段只允许在验证集调整这些参数；进入最终测试前应将配置快照写入报告。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PortfolioConfig:
    execution_rule_version: str = "open_limit_v2"
    topk: int = 50
    buffer: int = 30
    max_drop: int = 10
    rebalance_rule: str = "every_n_trading_days"
    rebalance_days: int = 5
    rebalance_offset: int = 0
    initial_cash: float = 100_000_000.0
    buy_cost: float = 0.0005
    sell_cost: float = 0.0015
    min_commission: float = 5.0
    lot_size: int = 100
    winsor_quantile: float = 0.01
    max_industry_weight: float = 0.20
    use_mean_variance_optimizer: bool = False
    max_stock_weight: float = 0.04
    covariance_lookback: int = 120
    covariance_min_observations: int = 40
    alpha_scale: float = 0.025
    risk_aversion: float = 0.50
    turnover_penalty: float = 0.003
    target_annual_volatility: float = 0.35
    volatility_lookback: int = 60
    use_dynamic_exposure: bool = False
    minimum_exposure: float = 0.60
    # 单次调仓双边成交额 / 调仓前组合权益的硬上限。
    max_rebalance_turnover: float = 0.15
    drawdown_tiers: tuple = ((-0.20, 0.90), (-0.30, 0.75), (-0.40, 0.60))
    benchmark: str = "SH000300"

    def to_dict(self) -> dict:
        return asdict(self)
