# QUANT-ASHARE：A股量化机器学习选股系统

一个面向 A 股日频横截面选股的研究项目：将 Qlib 行情、Tushare 财务/估值数据按 Point-in-Time 原则整理，构建候选因子，使用机器学习预测未来收益，并将预测分数转换为可交易组合。

## 研究主线

```text
数据准备
  → 270个候选因子库
  → 270 → 58 → 25个人工核心因子
  → DeepSeek增量因子挖掘与准入
  → 冻结25+5=30个因子
  → LightGBM滚动样本外预测
  → TopK/Buffer/Dropout低频调仓回测
```

额外分析：Ridge / LightGBM / PyTorch MLP 对比；5日、10日、20日预测期限对比；因子稳定性、增量贡献、成本、容量和风险暴露分析。

## 核心设计

- 主标签：`label_ret_5`，预测未来5个交易日收益；10日和20日用于敏感性分析。
- 主模型：LightGBM；Ridge 为线性基线，MLP 为非线性对照。
- 训练方式：Purged Walk-Forward，通常为750日训练、60日局部验证、20日预测步长。
- 组合方式：Top50，Buffer=30，限制单次主动换出，次日开盘成交。
- 调仓方式：低频/周频思想；实际间隔由 `rebalance_days` 控制，当前默认20个交易日。
- 基准：沪深300，代码 `SH000300`。
- 约束：停牌、涨跌停、100股整手、行业上限、换手限制和交易成本。

## DeepSeek 因子挖掘

DeepSeek 只提出新的因子假设，不直接修改人工25因子。候选依次经过：公式与字段校验、防未来数据泄漏、缺失率检查、训练期 RankIC、增量IC、与已有因子去冗余，以及10bps成本下的低频组合准入。

当前主要门槛：最大缺失率不超过15%；绝对 RankIC 至少0.01；增量绝对 RankIC 至少0.005；与已有因子的日度 IC 相关性不得达到0.70冗余阈值；净 Sharpe 至少0.5、年化净超额不低于0、最大回撤绝对值不超过0.5。通过者进入注册表，最终冻结5个增量因子。

需要配置 `DEEPSEEK_API_KEY`。API只负责候选生成，实际计算、筛选、比较和回测由本地代码完成；没有 API 时，其余数据、因子和模型流程仍可独立运行。

## 时间划分与研究纪律

| 数据集 | 时间 | 用途 |
|---|---|---|
| 训练集 | 2016-01-01 至 2023-04-30 | 因子统计与模型训练 |
| 验证集 | 2023-05-01 至 2025-01-01 | 因子、模型、超参数和组合参数选择 |
| 测试集 | 2025-01-02 至 2026-08-13 | 全部冻结后的最终评估，仅使用一次 |

财务数据按公告日对齐，标签列从特征中排除，滚动标签设置 purge/embargo。测试集不能用于反向挑选因子、模型或调仓参数。

## 快速开始

```powershell
python main.py list
```

主流程：

```powershell
python main.py fetch_daily
python main.py fetch_financial
python main.py merge_data
python main.py fetch_stock_basic
python main.py clean_data
python main.py fetch_daily_basic
python main.py build_alpha158
python main.py build_factors
python main.py evaluate_factors
python main.py remove_redundant
python main.py preprocess_factors
python main.py train_lgb --label label_ret_5 --eval-segments valid
python main.py portfolio_backtest --model lgb --label label_ret_5 --segment valid
```

三模型比较：

```powershell
python main.py train_ridge --label label_ret_5 --eval-segments valid
python main.py train_lgb --label label_ret_5 --eval-segments valid
python main.py train_mlp --label label_ret_5 --eval-segments valid
python main.py compare_models --label label_ret_5 --segment valid
```

DeepSeek 因子挖掘：

```powershell
python main.py qm_build_mining_memory
python main.py qm_mine_one_factor
```

## 主要目录

```text
config/                 日期、路径和策略配置
scripts/                数据获取、清洗和流水线脚本
factors/                自研因子注册表与算子
quantmind_integration/  DeepSeek候选解析、评估和准入
preprocessing/          去极值、中性化和标准化
models/                 LightGBM、Ridge、MLP和滚动训练
backtest/               信号处理、组合构建和回测引擎
analysis/               因子、模型、标签和结果分析
data/                   原始数据和处理后产物
reports/                实验报告和审计结果
tests/                  时间隔离、模型和回测测试
```

## 详细文档

- [PROJECT_INTRO.md](PROJECT_INTRO.md)：完整运行顺序、文件机制、经济逻辑、面试问答和优化方向；
- [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md)：项目总体蓝图；
- `main.py`：命令入口；
- `config/settings.py`：路径、日期和数据配置。

## 声明

本项目是历史数据研究工程，不构成投资建议，也不是实盘交易系统。回测结果受数据质量、样本区间、市场环境、交易成本和容量影响，不能代表未来收益。
