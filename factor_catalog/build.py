from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from config.settings import DATA_PROCESSED, ROOT

CATEGORY_DEFINITIONS = {
    "momentum_reversal": "反转与动量",
    "trend_quality": "趋势质量",
    "volatility_risk": "波动与风险",
    "liquidity": "流动性",
    "price_volume": "量价关系",
    "price_structure": "价格结构",
    "technical_state": "技术状态",
    "value": "估值",
    "financial_quality": "财务质量",
    "financial_growth_change": "财务成长与变化",
    "industry_market_state": "行业与市场状态",
    "market_sentiment": "市场情绪",
    "policy_text": "政策文本",
    "announcement_event": "事件公告",
}

CUSTOM_MODULE_CATEGORY = {
    "reversal_momentum": "momentum_reversal",
    "volatility": "volatility_risk",
    "liquidity": "liquidity",
    "technical": "technical_state",
    "value": "value",
    "quality": "financial_quality",
    "growth": "financial_growth_change",
    "price_structure": "price_structure",
    "volume_price": "price_volume",
}

CATEGORY_OVERRIDES = {
    "MAX_DD20": "volatility_risk",
    "MAX_DD60": "volatility_risk",
}

ALPHA_FAMILY_CATEGORY = {
    "ROC": "momentum_reversal",
    "MA": "trend_quality", "BETA": "trend_quality", "RSQR": "trend_quality", "RESI": "trend_quality",
    "STD": "volatility_risk",
    "MAX": "price_structure", "MIN": "price_structure", "QTLU": "price_structure",
    "QTLD": "price_structure", "RANK": "price_structure", "RSV": "price_structure",
    "IMAX": "price_structure", "IMIN": "price_structure", "IMXD": "price_structure",
    "CORR": "price_volume", "CORD": "price_volume", "WVMA": "price_volume",
    "CNTP": "trend_quality", "CNTN": "trend_quality", "CNTD": "trend_quality",
    "SUMP": "trend_quality", "SUMN": "trend_quality", "SUMD": "trend_quality",
    "VMA": "price_volume", "VSTD": "price_volume", "VSUMP": "price_volume",
    "VSUMN": "price_volume", "VSUMD": "price_volume",
}

CUSTOM_INPUTS = {
    "value": {"EP": ["pe_ttm"], "BP": ["pb"], "SP": ["ps_ttm"]},
    "quality": {
        "ROE": ["roe"], "ROA": ["roa"], "GPM": ["grossprofit_margin"],
        "NPM": ["netprofit_margin"], "CURRENT_RATIO": ["current_ratio"],
        "LOW_DEBT": ["debt_to_assets"], "ROE_WAA": ["roe_waa"], "ROIC": ["roic"],
        "QUICK_RATIO": ["quick_ratio"], "ASSETS_TURN": ["assets_turn"],
    },
    "growth": {
        "REV_YOY": ["tr_yoy"], "PROFIT_YOY": ["netprofit_yoy"],
        "REV_ACCEL": ["tr_yoy"], "PROFIT_ACCEL": ["netprofit_yoy"],
        "OP_YOY": ["op_yoy"], "OCF_YOY": ["ocf_yoy"], "ROE_YOY": ["roe_yoy"],
        "Q_SALES_YOY": ["q_sales_yoy"], "Q_OP_QOQ": ["q_op_qoq"],
    },
}

KNOWN_REVIEW = {
    "MOM20": {"status": "deprecated", "review_status": "merged", "alias_of": "internal.REV20", "alias_transform": "negate"},
    "MOM60": {"status": "deprecated", "review_status": "merged", "alias_of": "internal.REV60", "alias_transform": "negate"},
    "OVERNIGHT_RET5": {"status": "deprecated", "review_status": "merged", "alias_of": "internal.GAP5", "alias_transform": "negate"},
    "OVERNIGHT_RET10": {"status": "deprecated", "review_status": "merged", "alias_of": "internal.GAP10", "alias_transform": "negate"},
    "CONSEC_UP3": {"review_status": "fixed", "formula_revision": 2, "official_value_status": "replace_approved", "change_note": "改为截至当日、上限3日的连续上涨收益计数"},
    "CONSEC_UP5": {"review_status": "fixed", "formula_revision": 2, "official_value_status": "replace_approved", "change_note": "改为截至当日、上限5日的连续上涨收益计数"},
    "MAX_DD20": {"review_status": "fixed", "formula_revision": 2, "direction": "negative", "official_value_status": "not_replaced", "change_note": "改为遵守峰谷时间顺序的20日路径最大回撤；值越大代表风险越高"},
    "MAX_DD60": {"review_status": "fixed", "formula_revision": 2, "direction": "negative", "official_value_status": "not_replaced", "change_note": "改为遵守峰谷时间顺序的60日路径最大回撤；值越大代表风险越高"},
}


def _import_custom_factors() -> None:
    for module in [
        "reversal_momentum", "volatility", "liquidity", "technical", "value",
        "quality", "growth", "price_structure", "volume_price", "gp_huatai_replication",
    ]:
        __import__(f"factors.base.{module}")


def _module_name(func) -> str:
    return func.__module__.rsplit(".", 1)[-1]


def _custom_inputs(name: str, module: str) -> list[str]:
    explicit = CUSTOM_INPUTS.get(module, {}).get(name)
    if explicit:
        return explicit
    if module == "reversal_momentum":
        return ["open", "close"] if name.startswith(("INTRADAY", "OVERNIGHT")) else ["close"]
    if module == "volatility":
        return ["open", "high", "low", "close"] if "ATR" in name or "RANGE" in name else ["close"]
    if module == "liquidity":
        if name in {"LOW_TURNOVER", "LN_TURNOVER"}: return ["turnover_rate"]
        if name == "AMIHUD_20": return ["close", "amount"]
        if name == "VOL_TREND": return ["volume"]
        return ["close", "volume"]
    if module == "volume_price":
        if name.startswith("RET_AMT") or name.startswith("AMOUNT_TREND"): return ["close", "amount"]
        if name.startswith("VWAP_DEV"): return ["close", "vwap"]
        return ["close", "volume"]
    if module == "price_structure":
        if name.startswith(("GAP", "MA_ALIGN", "BIAS")): return ["open", "close"] if name.startswith("GAP") else ["close"]
        return ["open", "high", "low", "close"]
    if module == "technical":
        if name.startswith(("OBV",)): return ["close", "volume"]
        if name.startswith(("CCI", "KDJ", "WILLR", "BB_")): return ["high", "low", "close"]
        return ["close"]
    if module == "gp_huatai_replication": return ["open", "high", "low", "close", "volume", "vwap"]
    raise ValueError(f"未定义输入字段: {module}.{name}")


def _alpha158_expressions() -> dict[str, str]:
    path = ROOT / "vendor" / "quantmind" / "rd-agent" / "rdagent" / "utils" / "qlib.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "ALPHA158" for target in targets):
                value = ast.literal_eval(node.value)
                if len(value) != 158:
                    raise ValueError(f"Alpha158表达式数量异常: {len(value)}")
                return value
    raise ValueError("未在RD-Agent工具中找到ALPHA158表达式")


def _alpha_family(name: str) -> str:
    if name in {"KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2", "OPEN0", "HIGH0", "LOW0", "VWAP0"}:
        return "KBAR" if name.startswith("K") else "PRICE_RELATIVE"
    return re.sub(r"\d+$", "", name)


def _expression_inputs(expression: str) -> list[str]:
    return sorted(set(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", expression)))


def _window_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def _load_factor_sets() -> dict[str, set[str]]:
    result = {}
    for key, filename in {
        "stable_v2": "passed_factor_cols_v2.json", "passed_ret20": "passed_factor_cols_ret20.json",
        "passed_ret5": "passed_factor_cols.json",
    }.items():
        path = DATA_PROCESSED / filename
        result[key] = set(json.loads(path.read_text(encoding="utf-8")).get("factors", [])) if path.exists() else set()
    return result


def build_catalog() -> dict[str, Any]:
    import pyarrow.parquet as pq
    _import_custom_factors()
    from factors.registry import get_factor, get_factor_list

    physical = [name for name in pq.ParquetFile(DATA_PROCESSED / "factors.parquet").schema.names if name not in {"symbol", "date", "label_ret_5", "label_ret_10", "label_ret_20"}]
    alpha_file = [name for name in pq.ParquetFile(DATA_PROCESSED / "alpha158.parquet").schema.names if name not in {"symbol", "date"}]
    sets = _load_factor_sets()
    factors = []
    for name in get_factor_list():
        info = get_factor(name); module = _module_name(info["func"]); composite = module == "gp_huatai_replication"
        factors.append({
            "factor_id": f"internal.{name}", "physical_name": name, "layer": 3 if composite else 2,
            "factor_kind": "composite" if composite else "atomic", "source": "internal",
            "source_bundle": "custom", "implementation": f"{info['func'].__module__}:{getattr(info['func'], '__name__', '<callable>')}",
            "formula_modified": None, "primary_category": "composite" if composite else CATEGORY_OVERRIDES.get(name, CUSTOM_MODULE_CATEGORY[module]),
            "tags": [module, "daily"], "inputs": _custom_inputs(name, module),
            "lookback": _window_from_name(name), "formula_description": info["comment"],
            "availability": "after_close", "earliest_execution": "next_trade_day",
            "status": "experimental" if composite else "unreviewed", "review_status": "pending",
            "comment": info["comment"],
            "in_current_270": name in physical,
            "evaluation_membership": [key for key, names in sets.items() if name in names],
        })
        factors[-1].update(KNOWN_REVIEW.get(name, {}))
    expressions = _alpha158_expressions()
    for name, expression in expressions.items():
        family = _alpha_family(name)
        factors.append({
            "factor_id": f"qlib.alpha158.{name}", "physical_name": name, "layer": 2,
            "factor_kind": "atomic_engineered", "source": "microsoft_qlib",
            "source_bundle": "qlib_alpha158", "expression": expression,
            "formula_description": "保留 Qlib Alpha158 原始表达式；本项目不改写公式",
            "expression_source": "vendor/quantmind/rd-agent/rdagent/utils/qlib.py",
            "formula_modified": False,
            "primary_category": "price_structure" if family in {"KBAR", "PRICE_RELATIVE"} else ALPHA_FAMILY_CATEGORY[family],
            "tags": [family.lower(), "daily", "external"], "inputs": _expression_inputs(expression),
            "lookback": _window_from_name(name), "availability": "after_close",
            "earliest_execution": "next_trade_day", "status": "benchmark",
            "in_current_270": name in physical, "evaluation_membership": [key for key, names in sets.items() if name in names],
        })
    catalog = {
        "catalog_version": 1, "layer": 2, "name": "原子因子目录",
        "categories": CATEGORY_DEFINITIONS,
        "rules": {"single_primary_concept": True, "inputs_required": True, "availability_required": True,
                  "composite_expression_forbidden": True, "external_formula_immutable": True},
        "physical_factor_count": len(physical), "alpha158_physical_count": len(alpha_file),
        "factors": factors,
    }
    validate_catalog(catalog, physical)
    return catalog


def validate_catalog(catalog: dict[str, Any], physical: list[str] | None = None) -> None:
    ids = [item["factor_id"] for item in catalog["factors"]]
    if len(ids) != len(set(ids)): raise ValueError("factor_id存在重复")
    id_set = set(ids)
    for item in catalog["factors"]:
        if not item.get("inputs"): raise ValueError(f"{item['factor_id']}缺少inputs")
        if not item.get("availability") or not item.get("earliest_execution"): raise ValueError(f"{item['factor_id']}缺少时间可用性")
        if item["source_bundle"] == "qlib_alpha158" and item.get("formula_modified") is not False:
            raise ValueError(f"{item['factor_id']}未保持Alpha158原式")
        if item.get("alias_of") and item["alias_of"] not in id_set:
            raise ValueError(f"{item['factor_id']}的标准因子不存在: {item['alias_of']}")
    if physical is not None:
        represented = {item["physical_name"] for item in catalog["factors"] if item["in_current_270"]}
        if represented != set(physical):
            raise ValueError(f"物理因子覆盖不完整: missing={set(physical)-represented}, extra={represented-set(physical)}")


def main() -> int:
    catalog = build_catalog()
    output = ROOT / "config" / "factor_catalog.json"
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    layer2 = [item for item in catalog["factors"] if item["layer"] == 2 and item["in_current_270"]]
    counts = {key: sum(item["primary_category"] == key for item in layer2) for key in CATEGORY_DEFINITIONS}
    print(json.dumps({"physical_270": len(layer2), "catalog_identities": len(catalog["factors"]),
                      "deferred_layer3": sum(item["layer"] == 3 for item in catalog["factors"]),
                      "categories": counts}, ensure_ascii=False))
    print(f"[OK] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
