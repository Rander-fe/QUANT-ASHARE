from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from config.settings import ROOT, TEST_PERIOD, TRAIN_PERIOD

OPERATOR_CATALOG = ROOT / "config" / "operator_catalog.json"
MAPPING_FILE = ROOT / "config" / "quantmind_qlib_operator_mapping.json"
POLICY_FILE = ROOT / "config" / "quantmind_candidate_policy.json"
FINANCIAL_FIELDS = {
    "roe", "roe_waa", "roa", "roic", "grossprofit_margin", "netprofit_margin",
    "current_ratio", "quick_ratio", "debt_to_assets", "assets_turn", "tr_yoy",
    "netprofit_yoy", "op_yoy", "ocf_yoy", "roe_yoy", "q_sales_yoy", "q_op_qoq",
}

# 算子位置参数个数范围(min,max)。来源: factors/operators_v2.py 的函数签名。
# DSL 里 FIN_* 的第2个可见参数是报告期数(periods), 与底层签名(report_period)不同, 故单独给 1-2。
# CS_RANK/CS_ZSCORE 需要 (series, dates) 两列; 由于公式字段白名单不含 date, 实际不可表达, 保留 2 参约束以防误用。
OPERATOR_ARITY: dict[str, tuple[int, int]] = {
    "DELAY": (1, 2), "DELTA": (1, 2), "TS_PCT_CHANGE": (2, 2),
    "TS_MEAN": (2, 2), "TS_SUM": (2, 2), "TS_STD": (2, 2),
    "TS_MEDIAN": (2, 2), "TS_QUANTILE": (3, 3), "TS_MIN": (2, 2), "TS_MAX": (2, 2),
    "TS_SKEW": (2, 2), "TS_KURT": (2, 2), "TS_POSITION": (2, 2), "TS_RANK_PCT": (2, 2),
    "TS_CORR": (3, 3), "TS_COV": (3, 3), "TS_MAD": (2, 2), "EWM_MEAN": (2, 2),
    "TS_COUNT": (2, 2), "TS_CONSECUTIVE_COUNT": (2, 2), "TS_RSQUARE": (3, 3),
    "TS_SLOPE": (2, 2), "TS_DECAY_LINEAR": (2, 2), "SAFE_DIV": (2, 3),
    "CLIP": (3, 3), "ABS": (1, 1), "SIGN": (1, 1), "LOG1P": (1, 1), "SQRT": (1, 1),
    "WHERE": (3, 3), "FIN_LAG_REPORT": (1, 2), "FIN_DELTA_REPORT": (1, 2),
    "CS_RANK": (2, 2), "CS_ZSCORE": (2, 2),
}


def load_policy() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in
                 (OPERATOR_CATALOG, MAPPING_FILE, POLICY_FILE))


def audit_integration() -> dict[str, Any]:
    catalog, mapping, policy = load_policy()
    approved = {item["id"] for item in catalog["operators"] if item["status"] == "approved"}
    mapped = set(mapping["operators"])
    errors = []
    if approved != mapped:
        errors.append({"mapping_missing": sorted(approved - mapped), "mapping_extra": sorted(mapped - approved)})
    if policy["selection_period"] != list(TRAIN_PERIOD): errors.append("selection_period未锁定训练期")
    if policy["forbidden_selection_period"] != list(TEST_PERIOD): errors.append("测试期边界不一致")
    if set(policy["allowed_fields"]) & set(policy["forbidden_fields"]): errors.append("允许与禁止字段重叠")
    modes = {}
    for item in mapping["operators"].values(): modes[item["mode"]] = modes.get(item["mode"], 0) + 1
    return {"approved": len(approved), "mapped": len(mapped), "mapping_modes": modes, "errors": errors}


def _normalize_formula(formula: str) -> str:
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"FIELD_\1", formula)


def validate_candidate_batch(candidates: list[dict[str, Any]]) -> None:
    catalog, _, policy = load_policy()
    if not policy["factors_per_round"]["min"] <= len(candidates) <= policy["factors_per_round"]["max"]:
        raise ValueError("QUANTMIND每轮只能提交1至5个因子")
    allowed_ops = {item["id"] for item in catalog["operators"] if item["status"] == "approved"}
    allowed_fields = set(policy["allowed_fields"])
    required = set(policy["required_keys"])
    seen = set()
    for candidate in candidates:
        missing = required - set(candidate)
        if missing: raise ValueError(f"候选缺少元数据: {sorted(missing)}")
        name = candidate["factor_name"]
        if name in seen: raise ValueError(f"本轮因子名称重复: {name}")
        seen.add(name)
        inputs = set(candidate["inputs"])
        if not inputs <= allowed_fields: raise ValueError(f"{name}使用未批准字段: {sorted(inputs-allowed_fields)}")
        if candidate["availability"] != "after_close": raise ValueError(f"{name}可用时间必须为after_close")
        tree = ast.parse(_normalize_formula(candidate["formula"]), mode="eval")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"FIN_LAG_REPORT", "FIN_DELTA_REPORT"}):
                continue
            if len(node.args) not in (1, 2) or not isinstance(node.args[0], ast.Name):
                raise ValueError(f"{name}的{node.func.id}必须写成{node.func.id}($financial_field, n)")
            field = node.args[0].id.removeprefix("FIELD_")
            if field not in FINANCIAL_FIELDS:
                raise ValueError(f"{name}的{node.func.id}只允许财务字段，当前为{field}")
            if len(node.args) == 2 and not (
                isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, int)
                and not isinstance(node.args[1].value, bool) and node.args[1].value >= 1
            ):
                raise ValueError(f"{name}的{node.func.id}报告期数必须是正整数常量")
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        if not calls <= allowed_ops: raise ValueError(f"{name}使用未批准算子: {sorted(calls-allowed_ops)}")
        # 算子参数个数区间检查: 防止单序列算子被误当多序列使用(如TS_RSQUARE($close,20))导致求值崩溃。
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            op_id = node.func.id
            if op_id not in OPERATOR_ARITY:
                continue
            lo, hi = OPERATOR_ARITY[op_id]
            if not lo <= len(node.args) <= hi:
                raise ValueError(
                    f"{name}的{op_id}参数个数应为{lo}-{hi}个,当前{len(node.args)}个"
                )
        fields = {node.id.removeprefix("FIELD_") for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and node.id.startswith("FIELD_")}
        if fields != inputs: raise ValueError(f"{name}公式字段与inputs不一致: formula={sorted(fields)}, inputs={sorted(inputs)}")
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        unknown_names = names - calls - {f"FIELD_{field}" for field in inputs}
        if unknown_names: raise ValueError(f"{name}公式包含未知名称: {sorted(unknown_names)}")
        forbidden_nodes = (ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp, ast.DictComp, ast.GeneratorExp)
        if any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)):
            raise ValueError(f"{name}公式包含禁止语法")


def assert_selection_period(start: str, end: str) -> None:
    if [start, end] != list(TRAIN_PERIOD):
        raise ValueError(f"SOTA筛选只能使用训练期 {TRAIN_PERIOD[0]} 至 {TRAIN_PERIOD[1]}")


def main() -> int:
    result = audit_integration()
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__": raise SystemExit(main())
