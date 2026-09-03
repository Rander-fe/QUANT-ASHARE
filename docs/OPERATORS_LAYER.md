# 第1层：基础算子

第1层把经过第0层准入的数据字段转换为可组合的基础运算。算子不是选股因子，不携带“越大越好”的投资方向。

旧实现在 `factors/operators.py`，仅用于历史兼容。QUANTMIND安全实现在
`factors/operators_v2.py`，准入目录在 `config/operator_catalog.json`。

## 当前状态

- 已登记并实现22个安全算子；旧10个名称保留映射但不向QUANTMIND开放。
- 所有滚动算子要求完整窗口，不跨NaN填充。
- `TS_POSITION`与真正的`TS_RANK_PCT`已经分开。
- 时序算子通过`apply_by_symbol`强制股票边界；横截面算子按交易日分组。
- 历史行业不安全，因此行业排名、中性化和残差化暂缓开放。

运行审计：

```powershell
python main.py audit_operators
python -m unittest tests.test_operators_v2 -v
```

## 安全输入边界

- 暂不允许使用当前快照回填的`industry`。
- 估值字段进入算子前必须把目录声明的0哨兵恢复为NaN。
- 暂不允许使用空现金流数值表和空政策正文。
- 当日收盘字段生成的信号最早下一交易日执行。
