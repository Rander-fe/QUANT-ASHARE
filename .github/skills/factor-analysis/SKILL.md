---
name: factor-analysis
description: '分析A股因子的IC/ICIR/分组单调性/中性化/相关性。Use when: 计算因子IC、ICIR、因子回测、因子筛选、中性化、行业中性、市值中性、因子相关性、冗余因子、alphalens、jqfactor_analyzer。'
argument-hint: '因子名或因子列表'
---

# 因子分析

## 目标
用 IC/ICIR 及质量指标综合评估因子，筛选有效因子进入模型。

## 评估指标
- **IC 均值**：因子值与下期收益的秩相关（spearman）。
- **ICIR**：IC 均值 / IC 标准差。
- **t 值 / p 值**：IC 显著性。
- **10 分组单调性**：按因子值分 10 组，各组收益应单调。
- **换手率**：因子组合的换手成本。
- **因子相关性**：两两相关 >0.7 视为冗余，剔除其一。
- **中性化**：行业 / 市值中性化后的有效性。

## Procedure
1. 用 `jqfactor_analyzer` 或 `alphalens` 计算 IC 序列。
2. 输出 IC 均值、ICIR、t 值、p 值、IR 分层。
3. 生成 10 分组收益表与分层回测图。
4. 计算因子相关性矩阵，标记 >0.7 的冗余对。
5. 中性化后重新评估，保留稳健因子。

## 目标阈值
- IC ≥ 0.1，ICIR ≥ 0.08（ICIR 可达不到，如实记录）。

## 参考
- 指标目标见 [copilot-instructions.md](../../copilot-instructions.md)
