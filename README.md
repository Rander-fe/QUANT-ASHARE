# QUANT-ASHARE：A股量化机器学习选股研究系统

这是一个面向 A 股日频横截面选股的完整研究工程。项目把 Qlib 日线行情、Tushare 财务/估值数据和交易状态数据整理成 Point-in-Time（历史时点真实可见）样本，构建约270个候选因子，筛选人工核心因子，再利用机器学习预测股票未来收益，最后通过低频 TopK 组合进行交易回测。

项目重点不是“训练一个模型看收益”，而是完整检验：数据是否当时可见、因子是否有经济逻辑、模型是否有样本外预测能力、信号是否能扣除交易成本后执行。

## 一、完整实验流程

```text
数据准备
  → 270个候选因子库构建
  → 270个因子统计筛选为58个
  → 58个因子进一步筛选为25个人工核心因子
  → 以25因子为基准进行DeepSeek增量因子挖掘
  → 候选因子合法性、预测能力、增量性和交易准入
  → 冻结5个增量因子，组成25+5=30个最终因子
  → LightGBM滚动样本外预测
  → TopK/Buffer/Dropout低频组合回测
  → 额外分析：三模型、三种预测期限、稳定性与风险归因
```

上一阶段产物是下一阶段的输入，不能跳过。最终主模型以30个冻结因子和 `label_ret_5` 为准。

## 二、数据准备

### 2.1 数据来源

- Qlib：日线开盘价、最高价、最低价、收盘价、成交量、成交额、复权信息等；
- Tushare：利润表、资产负债表、现金流量表、财务指标、估值、行业、上市日期、历史名称和交易状态；
- 沪深300：代码 `SH000300`，作为组合相对收益基准。

### 2.2 数据处理逻辑

1. 获取沪深及北交所日线行情；
2. 拉取财务报表并生成以公告日为生效日的财务事件表；
3. 用 backward as-of join 将财务数据合并到行情日期，避免报告期数据提前使用；
4. 根据历史名称识别 ST、退市和特殊状态；
5. 清理重复记录、指数代码、上市不足60日股票和无效交易记录；
6. 保留停牌导致的缺失，不把无法交易的数据虚构成正常价格；
7. 合并行业、市值、估值、涨跌停和一字板等交易信息。

主要文件和产物：

```text
scripts/fetch_daily.py
scripts/fetch_financial.py
scripts/merge_data.py
scripts/fetch_stock_basic.py
scripts/clean_data.py
scripts/fetch_daily_basic_by_date.py
data/processed/financial/financial_events.parquet
data/processed/merged_data.parquet
data/processed/basic_cleaned_with_extra_by_date.parquet
```

## 三、270个候选因子库

项目将自研因子和 Qlib Alpha158 组合成约270个候选特征，主要经济类别包括：

- 价值：PE、PB、PS、股息和估值相对水平；
- 质量：ROE、ROA、盈利能力、现金流质量和经营效率；
- 成长：收入、利润、现金流和资产增长；
- 动量与反转：不同窗口收益、趋势、短期反转和价格结构；
- 波动与风险：历史波动、下行风险、特质波动和残差风险；
- 流动性与交易行为：成交量、成交额、换手率、量价关系和交易拥挤；
- 规模与市场特征：市值、流动性分层和市场相对表现；
- Qlib Alpha158：作为标准化外部因子基线。

典型处理流程：

```text
行情/财务/估值数据
  → 因子公式和时序窗口计算
  → 生成未来5/10/20日收益标签
  → 每日横截面去极值
  → 行业和市值中性化
  → 横截面标准化
  → 保存factors.parquet
```

主要文件：`factors/`、`scripts/build_factors.py`、`scripts/build_alpha158.py`、`preprocessing/preprocess_factors.py`。

## 四、270→58→25人工核心因子

### 4.1 统计筛选

每个因子按交易日计算与未来收益的 IC 和 RankIC，并汇总 IC 均值、ICIR、正 IC 比例、缺失率和稳定性。当前研究口径包括：

```text
|IC均值| >= 0.01
|ICIR| > 0.05
日度IC相关性 > 0.70 的同质因子去冗余
```

负 IC 不一定是坏因子，只要方向稳定，可以在分组解释和模型中记录方向。

### 4.2 58→25

对通过基础统计的58个因子进一步检查：

- 因子之间是否表达同一种经济逻辑；
- 日度 IC 序列是否高度相关；
- 五分组收益是否具有单调性；
- 高低组是否形成稳定多空收益；
- 因子是否缺失严重或换手过高；
- 是否覆盖不同经济来源，而不是全部集中在动量或规模。

主要文件：

```text
analysis/evaluate_factors.py
analysis/remove_redundant_factors.py
analysis/factor_quantile.py
config/human_core_25_v1.json
data/processed/selected_factor_cols.json
data/processed/passed_factor_cols.json
```

## 五、DeepSeek增量因子挖掘

DeepSeek 不负责重建270因子库，也不修改人工25因子。它以25个人工因子、历史失败记录、允许字段和算子、标签期限及已准入因子为依据，提出新的经济假设和公式。

完整准入漏斗：

```text
构建挖掘记忆
  → DeepSeek生成候选公式
  → JSON/字段/算子/时间契约校验
  → 缺失率和快速筛选
  → 训练期label_ret_5评估
  → 增量RankIC与已有因子去重
  → 10bps低频组合回测
  → 注册通过者并冻结5个增量因子
```

候选主要标准：最大缺失率不超过15%；绝对 RankIC 至少0.01；相对人工25因子的增量绝对 RankIC 至少0.005；与人工25及已准入因子的日度 IC 相关性不能达到0.70冗余阈值；10bps成本下净 Sharpe 至少0.5、年化净超额不低于0、最大回撤绝对值不超过0.5。

主要文件：

```text
scripts/build_factor_mining_memory.py
scripts/run_deepseek_factor_trial.py
scripts/mine_one_factor.py
quantmind_integration/
quantmind_pipeline/
config/quantmind_candidate_policy.json
config/quantmind_admitted_registry.json
reports/factor_mining_memory.json
reports/quantmind_trials/
```

需要设置 `DEEPSEEK_API_KEY`。API 只负责生成候选，本地代码负责计算、筛选、比较、回测和注册；没有 API 时，其余数据、因子和模型流程仍可运行。

## 六、最终30因子LightGBM模型

最终模型输入为：

```text
25个人工核心因子 + 5个DeepSeek准入增量因子 = 30个最终因子
```

LightGBM 的任务不是预测价格，而是为每个交易日的股票输出横截面 `pred` 分数。分数越高，表示模型认为该股票未来5日相对收益越强。

训练采用 Purged Walk-Forward：通常使用最近750个交易日训练、60个交易日局部验证、20个交易日向前预测，并按标签期限设置 purge/embargo。模型滚动步长与组合调仓频率是两个不同参数。

为什么选 LightGBM：Ridge 作为稳定、可解释的线性基线；MLP 作为复杂非线性对照；LightGBM 更适合结构化因子表格数据，能学习阈值和因子交互，且验证期的 IC、头尾区分和组合转化综合更好。

主要文件：`models/lightgbm/train.py`、`models/lightgbm/data.py`、`models/lightgbm/evaluate.py`、`models/lightgbm/optuna_search.py`、`models/rolling.py`。

### LightGBM评分的四个层次

项目不是把所有指标机械相加，而是让不同指标服务于不同阶段：

| 层次 | 指标 | 作用 |
|---|---|---|
| 训练损失 | MSE | 让树逐步减少预测误差 |
| 早停指标 | 验证集 MSE | 判断是否继续增加树 |
| 调参目标 | 验证集平均 RankIC | 选择最适合股票排序的参数 |
| 投资验收 | 净Sharpe、年化超额、最大回撤、换手、成本 | 判断能否转化为可交易组合 |

LightGBM由许多棵树接力学习：后面的树不断修正前面树没有解释好的残差，并自动学习因子阈值和交互关系。一次 Optuna trial 就是“提出参数 → 训练 → 验证集早停 → 输出预测 → 计算平均 RankIC → 返回目标值”。因此，MSE是训练函数，RankIC是调参目标，净Sharpe等是最终投资验收标准。完整公式和小白版解释见 [PROJECT_INTRO.md](PROJECT_INTRO.md) 第8章。

当前最终30因子、`label_ret_5` 实验的实际 Optuna 目标为：

```text
Score = 调参区间平均 RankIC + 0.05 × 扣10bps成本后的 Top10% 组合效用
```

项目另有多折稳健调参版本：`平均跨折RankIC - 0.50 × 跨折标准差 + 0.05 × 平均经济效用`。当前 `best_params_v2.json` 保存的是 `label_ret_20`、40因子候选结果，不是最终30因子5日实验的评分结果。详细核查见 PROJECT_INTRO.md 第8.11.1.1节。

## 七、预测评分标准

设股票预测分数为 (s_{i,t})，未来5日收益为 (r_{i,t})：

```text
Pearson IC  = Corr(s, r)
RankIC      = Corr(rank(s), rank(r))
RankICIR    = mean(RankIC) / std(RankIC)
Top-Bottom  = Top组平均收益 - Bottom组平均收益
```

组合层指标为：

```text
年化收益 = (期末净值 / 期初净值)^(1/年数) - 1
年化超额 = 策略年化收益 - 沪深300年化收益
Sharpe    = mean(日净收益) / std(日净收益) × √252
回撤     = 当前净值 / 历史最高净值 - 1
最大回撤 = 所有回撤中的最小值
双边换手 = (买入金额 + 卖出金额) / 期初资产
```

模型选择顺序是：先看验证期预测能力，再看稳定性和增量性，最后看扣成本后的组合结果。测试集只能在模型、因子和组合参数全部冻结后使用一次。

## 八、TopK/Buffer/Dropout低频调仓

LightGBM 输出分数后，组合模块在调仓日完成：

```text
预测分数
  → 输出端缩尾和行业/市值中性化
  → 按分数排序
  → 选择TopK
  → Buffer减少边界换手
  → Dropout/max_drop限制单次换出
  → 次日开盘成交
  → 扣除费用并计算净值
```

采用低频/周频思想，是因为日频排名会受到噪声影响，频繁交易会产生佣金、价差和冲击成本。Buffer体现“只有信号变化足够大才值得交易”；Dropout限制组合冲击；TopK在信号集中度、分散化、容量和成本之间折中。

当前默认配置由 `backtest/config.py` 控制，包括 Top50、Buffer=30、单次主动换出限制、行业上限、目标仓位和 `rebalance_days`。当前代码默认调仓间隔为20个交易日，报告必须以实际配置区分概念性“周频”和代码实现。

回测采用 T 日收盘形成信号、T+1 日开盘执行，处理停牌、涨跌停、整手、资金不足、换手上限、交易成本和沪深300基准。

主要文件：`backtest/alpha_signal.py`、`backtest/engine.py`、`backtest/run_backtest.py`、`backtest/config.py`、`analysis/tune_portfolio.py`。

## 九、额外分析

### 9.1 三模型对比

固定相同特征、标签、时间区间、purge和验证集，训练 Ridge、LightGBM、MLP，比较 Pearson IC、RankIC、RankICIR、Top-Bottom、换手和组合净绩效。最终选择 LightGBM 是综合考虑非线性能力、头尾收益、组合转化、稳定性和工程成本，不是机械选择单项 RankIC 最高者。

### 9.2 5日/10日/20日预测期限

```text
label_ret_5  → 未来5个交易日收益（主口径）
label_ret_10 → 未来10个交易日收益
label_ret_20 → 未来20个交易日收益
```

最终以5日为主，是因为人工25因子和DeepSeek准入已经以5日为主要口径，且更贴近短期相对强弱和低频信号更新。10日和20日用于检验信号是否跨期限稳定、长期标签是否更平滑、较低换手是否能抵消信号衰减。更换标签必须重新训练、设置对应 purge、生成预测和回测。

### 9.3 其他建议分析

- 25因子模型与25+5最终模型消融；
- 分年度、季度和牛熊市场状态的 RankIC；
- 不同 TopK、Buffer、调仓周期和成本的敏感性；
- 行业、市值、价值、动量、波动和流动性暴露归因；
- 不同随机种子、训练窗口和模型参数的稳定性；
- 成交量参与率、滑点和冲击成本压力测试；
- SHAP 等模型解释与外部 Alpha158 基线比较。

## 十、快速运行

```powershell
python main.py list

# 数据与因子
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

# 主模型与组合
python main.py train_lgb --label label_ret_5 --eval-segments valid
python main.py portfolio_backtest --model lgb --label label_ret_5 --segment valid

# 三模型对比
python main.py train_ridge --label label_ret_5 --eval-segments valid
python main.py train_mlp --label label_ret_5 --eval-segments valid
python main.py compare_models --label label_ret_5 --segment valid

# DeepSeek因子挖掘
python main.py qm_build_mining_memory
python main.py qm_mine_one_factor
```

## 十一、主要目录

```text
config/                 日期、路径、因子和策略配置
scripts/                数据获取、清洗和流程入口
factors/                自研因子与算子
quantmind_integration/  DeepSeek候选解析、评估和准入
preprocessing/          去极值、中性化和标准化
models/                 LightGBM、Ridge、MLP和滚动训练
backtest/               信号、组合、交易和回测引擎
analysis/               因子、标签、模型和结果分析
data/                   原始数据和处理产物
reports/                实验报告、因子记忆和审计结果
tests/                  时间隔离、模型和回测测试
```

完整的文件级运行逻辑、经济逻辑、面试问答和后续优化见 [PROJECT_INTRO.md](PROJECT_INTRO.md)，总体设计见 [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md)。

## 研究声明

本项目是历史数据研究工程，不构成投资建议，也不是实盘交易系统。回测结果受数据质量、样本区间、市场环境、交易成本、容量和实现细节影响，不能代表未来收益。
