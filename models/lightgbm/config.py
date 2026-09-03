# -*- coding: utf-8 -*-
"""LightGBM 滚动重训配置。

超参借鉴 Qlib 官方 benchmark（microsoft/qlib，examples/benchmarks 与
qlib/tests/config.py 中的 GBDT_MODEL，以及 examples/hyperparameter/LightGBM
的 Optuna 搜索区间），在此基础上结合本项目数据规模微调。

滚动窗口设计借鉴 Qlib 官方 Rolling Retrain（RR）基准：
    examples/benchmarks_dynamic/baseline/rolling_benchmark.py
    - 每 20 个交易日滚动重训一次
    - 每次用测试段之前最近 train_len 天训练、最近 valid_len 天做早停
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LightGBM 超参（qrun 基线 / qlib GBDT_MODEL 借鉴）
# ---------------------------------------------------------------------------
DEFAULT_LGB_PARAMS: dict = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.0421,          # qlib 基准优化值（原 0.0421）
    "colsample_bytree": 0.8879,       # 特征采样
    "subsample": 0.8789,              # 样本采样
    "subsample_freq": 1,
    "lambda_l1": 205.6999,            # L1 正则
    "lambda_l2": 580.9768,            # L2 正则
    "max_depth": 8,
    "num_leaves": 210,
    "min_data_in_leaf": 20,
    "num_threads": 8,                 # 按本机核数可调
    "verbosity": -1,
    "seed": 42,
}

# 训练轮数与早停
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# 滚动窗口（模型更新频率；与组合调仓频率相互独立）
# ---------------------------------------------------------------------------
ROLLING: dict = {
    # 每次滚动训练/生成预测前进的交易日数。它不是组合调仓周期；
    # 调仓周期由 backtest.config.PortfolioConfig.rebalance_days 单独控制。
    "step": 20,
    # 每次训练使用的历史交易日数（约 3 年）
    "train_len": 750,
    # 测试段之前用于早停的验证交易日数
    "valid_len": 60,
}

# ---------------------------------------------------------------------------
# 数据与标签
# ---------------------------------------------------------------------------
# 主标签：未来 5 个交易日收益率，与默认 5 日组合调仓周期一致。
LABEL_COL = "label_ret_5"
# 备选标签（可用 --label 切换）
ALT_LABEL_COLS = ("label_ret_5", "label_ret_10", "label_ret_20")

# 特征来源表：项目预处理产物（去极值+中性化+标准化后）
FACTORS_FILE = "features.parquet"
ALPHA158_FILE = None  # Alpha158 已并入 features.parquet（factors 表生成时合并）

# 精选因子清单：分层回测通过因子（多空>0 且 |单调性|>0.8，IC 符号对齐）
# 由 analysis/factor_quantile.py 生成。训练时优先按标签读取
# passed_factor_cols_ret5/ret10/ret20.json；本文件名仅用于旧产物兼容。
FACTOR_LIST_FILE = "passed_factor_cols.json"

# Optuna 搜索得到的最优超参（models/lightgbm/optuna_search.py 生成）
# train.py 若存在该文件则自动覆盖 DEFAULT_LGB_PARAMS
BEST_PARAMS_FILE = "best_params.json"

# 标签来源：预处理环节单独落盘的标签表
LABELS_FILE = "labels.parquet"

# 剔除指数 / B股代码（与 scripts/build_alpha158.py 保持一致）
NON_STOCK_PREFIXES = ("SH000", "SZ399", "SH900", "SZ200", "BJ")

# 主键列
KEY_COLS = ["symbol", "date"]

# features.parquet 中的非特征列（主键 + 交易状态标记）
# pct_chg 仅用于生成涨跌停标记，落盘时已剔除；此处双保险防止被当特征
NON_FEATURE_COLS = ("symbol", "date", "industry", "log_mv", "pct_chg",
                    "limit_up", "limit_down", "lock_limit_up", "lock_limit_down")
