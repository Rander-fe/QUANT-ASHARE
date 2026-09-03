"""Safe AST evaluator for the approved QUANTMIND factor DSL."""

from __future__ import annotations

import ast
import operator
import re

import numpy as np
import pandas as pd

from factors.operators_v2 import SAFE_OPERATORS


TIME_OPERATORS = {
    "DELAY", "DELTA", "TS_PCT_CHANGE", "TS_MEAN", "TS_SUM", "TS_STD",
    "TS_MEDIAN", "TS_QUANTILE", "TS_MIN", "TS_MAX", "TS_SKEW", "TS_KURT",
    "TS_POSITION", "TS_RANK_PCT", "TS_CORR", "TS_COV", "TS_MAD", "TS_COUNT",
    "TS_RSQUARE", "TS_SLOPE", "TS_DECAY_LINEAR",
}
FINANCIAL_REPORT_OPERATORS = {"FIN_LAG_REPORT", "FIN_DELTA_REPORT"}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow}
COMPARE = {ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _normalize(formula: str) -> str:
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"FIELD_\1", formula)


def _time_apply(name: str, args: list, frame: pd.DataFrame):
    func = SAFE_OPERATORS[name]
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for indices in frame.groupby("symbol", sort=False, observed=True).groups.values():
        local_args = [value.loc[indices] if isinstance(value, pd.Series) else value for value in args]
        result.loc[indices] = func(*local_args)
    return result


def _financial_report_apply(name: str, args: list, frame: pd.DataFrame):
    if "report_period" not in frame:
        raise KeyError(f"{name}需要report_period列；请先运行财报期索引与数据合并流程")
    if len(args) not in (1, 2):
        raise ValueError(f"{name}参数必须是(field)或(field, periods)")
    periods = args[1] if len(args) == 2 else 1
    if not isinstance(periods, int) or isinstance(periods, bool):
        raise ValueError(f"{name}的periods必须是正整数")
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for indices in frame.groupby("symbol", sort=False, observed=True).groups.values():
        result.loc[indices] = SAFE_OPERATORS[name](
            args[0].loc[indices], frame.loc[indices, "report_period"], periods
        )
    return result


def evaluate_formula(formula: str, frame: pd.DataFrame) -> pd.Series:
    """Evaluate an already policy-validated formula on symbol/date sorted data."""
    if not frame.sort_values(["symbol", "date"], kind="mergesort").index.equals(frame.index):
        raise ValueError("DSL输入必须按(symbol,date)排序并重建连续索引")
    tree = ast.parse(_normalize(formula), mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id.startswith("FIELD_"):
            column = node.id.removeprefix("FIELD_")
            if column not in frame:
                raise KeyError(f"缺少公式字段: {column}")
            return pd.to_numeric(frame[column], errors="coerce")
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            return BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            return COMPARE[type(node.ops[0])](visit(node.left), visit(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name, args = node.func.id, [visit(arg) for arg in node.args]
            if name not in SAFE_OPERATORS:
                raise ValueError(f"未批准算子: {name}")
            if name in FINANCIAL_REPORT_OPERATORS:
                return _financial_report_apply(name, args, frame)
            return _time_apply(name, args, frame) if name in TIME_OPERATORS else SAFE_OPERATORS[name](*args)
        raise ValueError(f"DSL不支持语法: {ast.dump(node, include_attributes=False)}")

    value = visit(tree)
    return value if isinstance(value, pd.Series) else pd.Series(value, index=frame.index, dtype="float64")
