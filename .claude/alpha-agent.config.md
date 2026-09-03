# Alpha Agent 配置（VernonOY/alpha-skills）

本文件被 alpha-* 系列技能（discover / evaluate / mine / library / backtest / monitor / report / signal / autopilot）读取。
本项目为 A股 选股，数据经 tushare 拉取并缓存到 `data_cache/`；qlib 二进制数据另存于 `~/.qlib/qlib_data/cn_data`（用于 backtest-qlib 技能）。

## 市场 / Market
MARKET: A-share           # A-share / HK / US

## 数据源 / Data Source
DATA_SOURCE: tushare      # tushare（默认，A股）/ csv（本地文件）
DATA_MODULE:              # 留空用 tushare 默认；或填写自定义模块路径
DATA_DIR: data/processed  # 仅 DATA_SOURCE=csv 时生效

## 语言 / Language
LANGUAGE: zh              # 报告/输出语言：zh / en

## 评估默认值 / Evaluation Defaults
# HOLDING_PERIODS: [5, 10, 20]
# IC_THRESHOLD: 0.02
# STRONG_ICIR: 0.5
# MODERATE_ICIR: 0.3

## Autopilot / 自动研究
# TARGET_LIBRARY_SIZE: 5
# MINING_CANDIDATES: 50
# QUALITY_THRESHOLD: moderate
# CORRELATION_THRESHOLD: 0.7
# AUTO_RETIRE_ON_ALERT: true
# MONITORING_WINDOW: 60
