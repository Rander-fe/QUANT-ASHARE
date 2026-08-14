---
name: backtest-qlib
description: '用Qlib LightGBM进行A股机器学习选股回测。Use when: 回测、机器学习选股、LightGBM、qrun、TopkDropoutStrategy、滚动训练、组合构建、收益率预测、调仓。'
argument-hint: '配置文件名或策略参数'
---

# Qlib 机器学习回测

## 目标
用 Qlib + LightGBM 训练收益率预测模型，构建组合并回测。

## 数据划分（铁律）
- **训练集**：2016 ~ 2023-05，滚动训练（rolling retraining）。
- **验证集**：2023-05 ~ 2025-01，仅用于选超参/模型/因子。
- **测试集**：2025-01 之后，**只最终评估一次**，严禁用测试集反向调方案。

## Procedure
1. 准备 Qlib 数据（见 data-ingest skill）。
2. 写 `configs/` 下的 workflow yaml（模型、数据、策略）。
3. `qrun configs/lightgbm.yaml` 跑基线。
4. 用 `TopkDropoutStrategy` 构建组合，设定调仓频率与持股数量。
5. 输出回测指标：年化、最大回撤、夏普、超额收益、IC/ICIR。
6. 记录结果到 `reports/`。

## 关键点
- 特征/标签对齐避免未来函数（特征过去、标签未来）。
- 滚动训练窗口与调仓频率保持一致。
- 验证集只用于调参，最终模型在测试集只评估一次。

## 参考
- 集划分铁律见 [copilot-instructions.md](../../copilot-instructions.md)
- Qlib 基线：`benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
