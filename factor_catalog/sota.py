"""生成并审计人工最终正式SOTA因子集合。"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from config.settings import DATA_PROCESSED, ROOT

LEGACY_SELECTION = DATA_PROCESSED / "selected_factor_cols.json"
CATALOG = ROOT / "config" / "factor_catalog.json"
OUTPUT = ROOT / "config" / "human_final_sota_v1.json"
OFFICIAL_LIBRARY = DATA_PROCESSED / "factors.parquet"
V2_LIBRARY = DATA_PROCESSED / "reversal_momentum_fixes_v2.parquet"


def build_final_sota() -> dict:
    selected = json.loads(LEGACY_SELECTION.read_text(encoding="utf-8"))["factors"]
    if len(selected) != 58 or len(set(selected)) != 58:
        raise ValueError("原人工筛选清单不是58个唯一因子")
    final_names = [
        "GAP5" if name == "OVERNIGHT_RET5" else
        "MAX_DD60_V2" if name == "MAX_DD60" else name
        for name in selected
    ]
    if len(final_names) != 58 or len(set(final_names)) != 58:
        raise ValueError("替换后不是58个唯一因子")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_name = {item["physical_name"]: item for item in catalog["factors"] if item["layer"] == 2}
    official_cols = set(pq.read_schema(OFFICIAL_LIBRARY).names)
    v2_cols = set(pq.read_schema(V2_LIBRARY).names)
    entries = []
    for name in final_names:
        if name == "MAX_DD60_V2":
            if name not in v2_cols: raise ValueError("V2库缺少MAX_DD60_V2")
            entries.append({
                "factor_name": name, "factor_id": "internal.MAX_DD60_V2",
                "category": "volatility_risk", "direction": "negative",
                "member_status": "provisional", "collection_role": "official_sota_member",
                "data_file": "data/processed/reversal_momentum_fixes_v2.parquet",
                "data_column": name, "formula_revision": 2,
                "note": "替代旧MAX_DD60；集合是最终正式SOTA，但该成员按用户要求保留provisional标记",
            })
            continue
        if name not in official_cols: raise ValueError(f"正式库缺少{name}")
        meta = by_name.get(name)
        if meta is None: raise ValueError(f"目录缺少{name}")
        entry = {
            "factor_name": name, "factor_id": meta["factor_id"],
            "category": meta["primary_category"], "direction": meta.get("direction", "learned"),
            "member_status": "active", "collection_role": "official_sota_member",
            "data_file": "data/processed/factors.parquet", "data_column": name,
        }
        if name == "CONSEC_UP5":
            entry.update({"formula_revision": 2, "note": "使用已正式写入因子库的修正公式值"})
        elif name == "GAP5":
            entry["note"] = "替代弃用的OVERNIGHT_RET5；二者方向相反，保留标准定义GAP5"
        entries.append(entry)

    result = {
        "sota_id": "human_final_sota_v1", "display_name": "人工优选58因子最终正式SOTA",
        "collection_status": "official_final_sota", "factor_count": 58,
        "selection_origin": "IC/ICIR筛选 + 日频IC相关性去冗余",
        "selection_parameters_legacy": {
            "min_abs_ic": 0.01, "min_icir": 0.05, "redundancy_threshold": 0.7,
        },
        "benchmark_sources": [{
            "benchmark_id": "qlib_alpha158",
            "role": "baseline_reference",
            "factor_count": 158,
            "source_bundle": "qlib_alpha158",
            "data_file": "data/processed/alpha158.parquet",
            "formula_modified": False,
            "automatic_sota_membership": False,
            "comparison_policy": "分别评估Alpha158基准与人工58 SOTA；同名因子不得重复入模",
        }],
        "immutability": {"quantmind_may_modify_members": False,
                         "quantmind_may_generate_candidates": True,
                         "generated_candidate_default_status": "experimental"},
        "quantmind_generation_policy": {
            "min_factors_per_round": 1,
            "max_factors_per_round": 5,
            "default_status": "experimental",
            "direct_sota_admission": False,
            "simple_factors_first": True,
            "may_modify_human_sota": False,
            "may_modify_alpha158": False,
            "required_metadata": [
                "factor_name", "description", "formula", "inputs", "lookback",
                "availability", "direction", "economic_rationale"
            ],
        },
        "replacements": {
            "OVERNIGHT_RET5": "GAP5", "MAX_DD60": "MAX_DD60_V2",
            "CONSEC_UP5": "same_name_formula_revision_2",
        },
        "factors": entries,
    }
    validate_final_sota(result)
    return result


def validate_final_sota(sota: dict) -> None:
    entries = sota["factors"]
    names = [item["factor_name"] for item in entries]
    if sota["collection_status"] != "official_final_sota": raise ValueError("集合不是最终正式SOTA")
    if len(entries) != 58 or len(names) != len(set(names)): raise ValueError("SOTA必须包含58个唯一因子")
    if "OVERNIGHT_RET5" in names or "MAX_DD60" in names: raise ValueError("旧因子仍在SOTA中")
    if not {"CONSEC_UP5", "GAP5", "MAX_DD60_V2"}.issubset(names): raise ValueError("指定修正因子缺失")
    max_dd = next(item for item in entries if item["factor_name"] == "MAX_DD60_V2")
    if max_dd["category"] != "volatility_risk" or max_dd["direction"] != "negative":
        raise ValueError("MAX_DD60_V2分类或方向错误")
    if max_dd["member_status"] != "provisional": raise ValueError("MAX_DD60_V2未保持provisional")
    benchmarks = sota.get("benchmark_sources", [])
    alpha = next((item for item in benchmarks if item.get("benchmark_id") == "qlib_alpha158"), None)
    if alpha is None or alpha.get("factor_count") != 158 or alpha.get("formula_modified") is not False:
        raise ValueError("Alpha158基准来源未完整保留")
    if alpha.get("automatic_sota_membership") is not False:
        raise ValueError("Alpha158不得自动并入人工58 SOTA")
    policy = sota.get("quantmind_generation_policy", {})
    if policy.get("min_factors_per_round") != 1 or policy.get("max_factors_per_round") != 5:
        raise ValueError("QUANTMIND每轮必须生成1至5个实验因子")
    if policy.get("default_status") != "experimental" or policy.get("direct_sota_admission") is not False:
        raise ValueError("QUANTMIND候选必须先进入experimental")


def main() -> int:
    sota = build_final_sota()
    OUTPUT.write_text(json.dumps(sota, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": sota["collection_status"],
                      "factor_count": sota["factor_count"], "provisional_members": [
                          x["factor_name"] for x in sota["factors"] if x["member_status"] == "provisional"
                      ]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
