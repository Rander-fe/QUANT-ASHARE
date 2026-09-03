# 第2层：原子因子

第2层目录位于 `config/factor_catalog.json`，由 `python main.py build_factor_catalog` 自动生成。

## 数量口径

- 当前 `factors.parquet` 有270个物理因子：112个自研原子因子 + 158个Alpha158。
- 当前代码注册表还有6个华泰GP复合因子，但它们不在现有270列中，目录将其标记为第3层。
- 因此目录共登记276个因子身份，第2层物理口径仍严格为270。

## 规则

- 每个原子因子只有一个主分类，但可以有多个标签。
- 必须登记输入字段、信息可用时间和最早执行时间。
- Alpha158保留原始表达式、来源与 `formula_modified=false`。
- 外部因子不改名覆盖；自研和外部因子使用不同命名空间。
- 复合GP或未来LLM组合公式不进入本层。

## 第一轮整理：反转与动量

- `CONSEC_UP3/5` 已修正为“截至当日的连续上涨天数”，不再把窗口内上涨收益错误相加。
- `MAX_DD20/60` 已修正为遵守峰值先于谷值的路径最大回撤。
- `MOM20/60` 保留历史列但标记为弃用别名，标准因子分别为 `REV20/60`，转换关系为取负。
- `OVERNIGHT_RET5/10` 保留历史列但标记为弃用别名，标准因子分别为 `GAP5/10`，转换关系为取负。
- 修正仅作用于后续重新计算；现存 `factors.parquet` 的历史值不会被静默覆盖。

## 审计

```powershell
python main.py build_factor_catalog
python main.py audit_factor_catalog
python -m unittest tests.test_factor_catalog -v
```
