# A股量化机器学习选股 · 项目总蓝图

> 本文档是整条流水线的「总蓝图 / 单一事实来源」。所有环节（数据 → 预处理 → 因子 → 建模 → 回测 → 分析）的实现必须与本文档及 `config/settings.py` 保持一致。
>
> 铁律（详见 `.github/copilot-instructions.md`）：**严禁未来函数**、**严禁用测试集反向选择方案**、**路径统一从 `config/settings.py` 读取**。
>
> 每个环节的**流程 / 关键参数 / 产物 / 数值**详见 `PROJECT_INTRO.md`（项目完整技术介绍）。

---

## 0. 一句话目标

> 构建一个 ≥100 因子的高质量因子库（基础 + 复合），经严格防未来函数的预处理后，用 LightGBM（基线）+ 可扩展模型预测 A 股未来收益率，按 TopK-Dropout 构建组合并回测；全程用验证集做因子/超参/模型选择，测试集只在最终评估时使用一次。

---

## 1. 铁律与时间划分（不可违反）

| 集合 | 区间 | 用途 | 约束 |
|------|------|------|------|
| 训练集 | 2016-01-01 ~ 2023-04-30 | 滚动训练（rolling retraining） | 可反复使用 |
| 验证集 | 2023-05-01 ~ 2025-01-01 | 仅用于超参 / 模型 / 因子选择 | 可反复使用 |
| 测试集 | 2025-01-02 ~ 2026-08-13 | 最终评估 | **只用一次** |

- **特征只用过去数据**：`Ref(x, d)`，d 为正。
- **标签只用未来数据**：`Ref(x, -d)`，d 为负。
- **财务数据必须 point-in-time**：以公告日（`ann_date` / `f_ann_date`）对齐，严禁用报告期 `end_date` 当可用日（未来函数）。
- 测试集若被污染，方案必须废弃重来——没有「看一眼再改」的空间。

---

## 2. 目标指标（KPI）

| 指标 | 目标 | 备注 |
|------|------|------|
| IC（RankIC 均值） | ≥ 0.10 | 首要目标 |
| ICIR（年化） | ≥ 0.08 | 可不达标，如实记录 |
| 因子质量 | 综合评估 | IC均值 / ICIR / t值 / p值 / 10分组单调性 / 换手率 / 相关性(>0.7剔除) / 行业·市值中性化后有效性 |
| 组合 | 超额收益 | 相对沪深300（SH000300），控制回撤与换手 |

---

## 3. 目录结构与职责

```
QUANT-ASHARE/
├── PROJECT_BLUEPRINT.md          # 本文档：总蓝图
├── main.py                       # ★ 流水线主程序（单环节跑 + 一键全流程）
├── config/
│   ├── settings.py               # ★ 路径 / 时间划分 / 数据源字段（唯一常量源）
│   └── __init__.py
├── data/
│   ├── raw/                      # 原始数据（财务在 raw/financial/）
│   └── processed/                # 处理后数据 & 因子表（parquet/csv）
├── factors/                      # ★ 因子库（基础 + 复合），注册表统计数量
│   ├── base/                     # 基础因子（价格/量/波动/财务等）
│   ├── composite/                # 复合因子（多基础因子合成）
│   └── registry.py               # 因子注册表 + 计数
├── preprocessing/                # 数据清洗 / point-in-time 对齐 / 特征-标签构造
├── models/                       # 模型定义与训练（LightGBM 基线 + 扩展）
├── backtest/                     # 组合构建与回测（TopK-Dropout + 仓位/择时）
├── analysis/                     # IC/ICIR/分组/中性化/相关性 分析
├── reports/                      # 回测结果 / 报告 / 图表
├── scripts/                      # 各环节独立可运行入口（含 main()）
├── knowledge/                    # 领域知识库（因子分类/评估方法/风险等）
└── mlruns/                       # MLflow 实验记录（自动生成）
```

### 路径常量（已在 `config/settings.py` 注册）
- `DATA_RAW` / `DATA_PROCESSED` / `FACTORS_DIR` / `MODELS_DIR` / `REPORTS_DIR`
- `FIN_RAW_DIR` / `FIN_PROCESSED_DIR`（财务专用）
- **新增目录前，先在 `settings.py` 注册常量，禁止脚本内硬编码绝对路径。**

---

## 4. 环节一：数据拉取（`scripts/`）

| 数据 | 来源 | 落盘 | 状态 |
|------|------|------|------|
| 行情/量价 | qlib 二进制库 `~/.qlib/qlib_data/cn_data` | 直接用 | ✅ 已就绪（6135 只，2000-2026） |
| 财务三大报表 | tushare | `data/raw/financial/*.parquet` | ✅ `scripts/fetch_financial.py` |
| 指数成分/行业 | tushare / qlib | `data/processed/` | ⬜ 待补（中性化需要） |

- 财务拉取按「报告期」批量，保留 `ann_date`/`f_ann_date` 供 point-in-time。
- 上游产物作为下游输入复用，避免重复拉取。

---

## 5. 环节二：因子库（`factors/`）—— 目标 ≥100 个

### 5.1 设计规范（硬性）
- 命名：`算子 + 窗口`，如 `ROC10`；因子带英文注释说明含义/出处。
- **去量纲**：除以 `$close` 或 `($volume + 1e-12)`，消除价格水平差异。
- **防除零**：分母统一 `+1e-12`。
- **多尺度**：同一算子跑多窗口（5/10/20/30/60 日）。
- **防未来函数**：特征 `Ref(x, d)`（d>0），标签 `Ref(x, -d)`（d<0）。

### 5.2 因子分类（种类不限，建议覆盖 8 大类）

> 依据 `knowledge/factor-taxonomy.md`，A股核心 alpha 来源是**反转**，动量弱/反转强，需综合多类。

| 类别 | 典型因子 | A股特征 |
|------|---------|---------|
| 1. 反转/均值回归 | 短期反转(5d)、RSI 反转、布林带回归、最大回撤回归 | **主导 alpha** |
| 2. 动量 | 动量(20d)、跳1月动量、残差动量、风险调整动量 | 弱/反转 |
| 3. 波动率 | 已实现波动率、下行波动、特质波动、偏度 | 低波动溢价 |
| 4. 量价/流动性 | 换手率、量比、Amihud 非流动性、OBV | 强 |
| 5. 财务质量 | ROE、毛利率、净利率、盈利增长、经营现金流/利润 | 需 point-in-time |
| 6. 估值 | PE、PB、PS、EV/EBITDA、股息率 | 价值溢价 |
| 7. 成长 | 营收/净利同比、加速度 | 需 point-in-time |
| 8. 技术/形态 | 均线乖离、MACD、KDJ、CCI、通道突破 | 辅助 |

### 5.3 基础因子 vs 复合因子
- **基础因子**：单一算子（如 `ROC10`、`STD20`、`VOL20`）→ 贡献约 60~70 个。
- **复合因子**：多个基础因子经 z-score / 等权 / IC 加权 / PCA 合成 → 贡献约 30~40 个。
  - 例：`MOM_VOL = z(ROC20) / (z(STD20) + 1e-12)`（风险调整动量）
  - 例：`REV_LIQ = -z(REV5) * z(TURN20)`（反转×流动性）
  - 例：`QUALITY = z(ROE) + z(毛利率) + z(经营现金流/利润)`

### 5.4 因子注册表（`factors/registry.py`）
- 统一注册所有因子，输出数量统计，供「是否达 100 个」检查与主程序编排。

---

## 6. 环节三：数据预处理（`preprocessing/`）

1. **数据清洗**：停牌/涨跌停/新股(ST)处理、极值 winsorize（如 3σ 或 5%/95%）、缺失值处理。
2. **Point-in-time 财务对齐**：财务值按公告日 `ann_date` 前推，严禁用报告期当天。
3. **特征构造**：所有因子在截面用**过去数据**（`Ref(x, d)`），形成特征矩阵。
4. **标签构造**：未来 N 日收益率 `label = Ref(close, -N) / close - 1`（如 N=5 或 20）。
5. **中性化（可选）**：行业中性 + 市值中性（回归取残差），输出中性化前后因子。
6. **落盘**：`data/processed/features.parquet`、`data/processed/labels.parquet`。

---

## 7. 环节四：机器学习建模（`models/`）

### 7.1 基线（已验证路径）
- **LightGBM**（`qlib.contrib.model.gbdt.LGBModel`），`loss=mse` 预测收益率。
- 滚动训练（rolling retraining），验证集做 early stopping。
- 特征：Alpha158（起步）→ 替换为自建因子库（第 5 节）。

### 7.2 可扩展
- XGBoost、NN（Qlib DNN）作为对比。
- 超参用验证集网格/贝叶斯调优（如 `optuna`），记录到 MLflow。

### 7.3 选型纪律
- **所有模型/超参/因子选择只看验证集指标**；测试集不参与任何选择。
- 复杂模型必须同时对比训练期方向对齐等权、训练期 IC 加权和 Ridge；缺少简单基准的模型比较无效。
- 正式比较统一生成实验 manifest，记录数据指纹、特征清单、时间区间、参数、代码版本和测试集使用状态。
- 事件/财务特征须满足 `available_date <= signal_date < execution_date < label_end_date`；缺少可用日字段时不得进入正式回测。
- 正式预处理缺少行业/市值/交易状态底表时必须失败，禁止静默生成降级研究产物。

---

## 8. 环节五：组合构建与回测（`backtest/`）

| 参数 | 基线值 | 说明 |
|------|--------|------|
| 策略 | `TopkDropoutStrategy` | 选 Top-K 只，剔除换出 N 只 |
| topk | 50 | 持股数量 |
| n_drop | 5 | 每次调仓换手 |
| 调仓频率 | 月频（20 交易日） | 可调周频/双周 |
| 账户 | 1e8 | |
| 基准 | SH000300 | 沪深300 |
| 成本 | open 0.0005 / close 0.0015 / min 5 | 可加滑点 |

### 8.1 仓位控制 / 择时 / 因子权重（可发挥项）
- **仓位控制**：按信号强度分档仓位；按波动率目标（vol targeting）缩放。
- **择时**：大盘均线 / 市场波动率 / 情绪指标决定总仓位 0%~100%。
- **因子权重调整**：ICIR 加权 / 动态权重（滚动 IC 衰减加权），定期再平衡。
- **约束**：行业/市值暴露约束、单票权重上限、换手率约束。

---

## 9. 环节六：分析评估（`analysis/`）

- **IC/ICIR**：RankIC 时序均值、年化 ICIR、t 值、p 值。
- **10 分组单调性**：分组收益单调、多空收益。
- **换手率**：调仓换手成本评估。
- **相关性**：因子间 >0.7 冗余剔除。
- **中性化后有效性**：行业/市值中性后 IC 是否仍显著。
- 输出报告与图表到 `reports/`。

---

## 10. 主程序编排（`main.py`）

支持两种方式：

```bash
# 单环节单独跑
python main.py --stage data          # 拉数据
python main.py --stage factors       # 算因子
python main.py --stage preprocess    # 预处理
python main.py --stage train         # 训练模型
python main.py --stage backtest      # 回测
python main.py --stage analyze       # 分析评估

# 一键全流程
python main.py --all
```

- 每个环节在 `scripts/` 或对应模块（`factors/` `models/`）暴露 `main()` + `if __name__ == "__main__":`。
- 新增环节时同步更新 `main.py` 的 stage 注册。

---

## 11. 执行环境（已就绪，勿改动）

- **解释器**：`C:/Users/haoran/miniconda3/envs/rqalpha/python.exe`（Python 3.9.25，qlib 0.9.7 唯一兼容环境）。
- 运行脚本永远用完整路径：
  ```bash
  C:/Users/haoran/miniconda3/envs/rqalpha/python.exe scripts/xxx.py
  ```
- qlib 0.9.7 两个已知坑（`record_temp` 过时 API、gym 刷屏）已在 `scripts/lgb_baseline.py` 内用猴子补丁处理，新脚本需复用同样补丁。

---

## 12. 实施路线（建议顺序）

1. ✅ 数据就绪（行情 + 财务）
2. ⬜ 补行业/成分数据（中性化需要）
3. ⬜ `factors/base/` 基础因子（60~70 个）
4. ⬜ `factors/composite/` 复合因子（30~40 个）→ 合计 ≥100
5. ⬜ `preprocessing/` 预处理 + point-in-time 对齐
6. ⬜ `analysis/` 因子评估（IC/ICIR/分组/中性化/相关性）→ 筛选高质量因子
7. ⬜ `models/` LightGBM 训练（验证集调优）
8. ⬜ `backtest/` 组合回测 + 仓位/择时/权重调整
9. ⬜ `main.py` 编排全流程
10. ⬜ 最终测试集评估（只用一次）

---

## 13. 关键风险与纪律提醒

- **未来函数**（10 种形态）：见 `knowledge/risk-and-pitfalls.md`，财务对齐尤其要 point-in-time。
- **幸存者偏差**：样本池需含退市股（qlib 库已含历史成分）。
- **过拟合**：验证集反复调优后，测试集退化为「第二次验证」——保持测试集纯净。
- **多重检验**：100+ 因子会有「假显著」，需 t 值/显著性校正。
- **成本与换手**：A股涨跌停、T+1、冲击成本见 `knowledge/market-microstructure.md`。
