每轮可输出1至5个候选因子，每个候选使用一个JSON成员。每个JSON成员必须包含：

```json
{
  "FACTOR_NAME": {
    "description": "因子类型与简要说明",
    "formula": "仅使用批准字段与32个白名单算子的安全DSL公式",
    "formulation": "与formula相同",
    "variables": {"close": "收盘价"},
    "inputs": "close（多个字段用英文逗号分隔）",
    "lookback": "20",
    "availability": "after_close",
    "direction": "positive|negative|learned",
    "economic_rationale": "可检验的经济逻辑"
  }
}
```

预测目标固定为未来5个交易日收益 `label_ret_5`。禁止修改人工核心25 SOTA和Alpha158；所有新因子默认experimental。禁止使用测试期选择因子。

生成链只提出新的经济逻辑，不在同一轮进行窗口微调。优化链只能依据既有回测反馈，对已登记逻辑做参数优化。
先读取结构化成功/失败经验，禁止重复完全相同、代数等价或与核心25处于高相关红海的公式。
注意：为兼容RD-Agent原生JSON类型约束，`inputs`与`lookback`必须先输出为字符串；本地适配层会转换并严格校验。
