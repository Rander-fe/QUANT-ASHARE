"""Distill QUANTMIND trial artifacts into compact generation memory."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "reports/quantmind_trials"
OUTPUT = ROOT / "reports/factor_mining_memory.json"


# 经济大类标签:输入字段 -> 大类。用于统计各方向探索进度与成色，指导禁区管理。
FIELD_CATEGORY = {
    "open": "price_structure", "high": "price_structure", "low": "price_structure",
    "close": "price_structure", "vwap": "price_structure", "adjclose": "price_structure",
    "pct_chg": "momentum", "volume": "liquidity_volume", "amount": "liquidity_volume",
    "turnover_rate": "liquidity_turnover", "total_mv": "size", "circ_mv": "size",
    "pe_ttm": "valuation", "pb": "valuation", "ps_ttm": "valuation",
    "roe": "profitability", "roe_waa": "profitability", "roa": "profitability",
    "roic": "profitability", "grossprofit_margin": "profitability", "netprofit_margin": "profitability",
    "current_ratio": "solvency", "quick_ratio": "solvency", "debt_to_assets": "solvency",
    "assets_turn": "operating", "tr_yoy": "growth", "netprofit_yoy": "growth",
    "op_yoy": "growth", "ocf_yoy": "growth", "roe_yoy": "growth",
    "q_sales_yoy": "growth", "q_op_qoq": "growth",
    "IND_REL_RET_20": "industry", "IND_RESID_RET_1D": "industry", "IND_RESID_MOM_20": "industry",
}
CATEGORY_LABEL = {
    "price_structure": "价量结构", "momentum": "动量", "liquidity_volume": "流动性/成交量",
    "liquidity_turnover": "流动性/换手", "size": "市值", "valuation": "估值",
    "profitability": "盈利质量", "solvency": "偿债/杠杆", "operating": "营运",
    "growth": "成长", "industry": "行业相对/残差",
}


def category_of(inputs: list[str]) -> str:
    for field in inputs:
        cat = FIELD_CATEGORY.get(field)
        if cat is not None:
            return cat
    return "unknown"


def main() -> int:
    records = []
    for candidate_path in sorted(TRIALS.glob("*/candidate.json")):
        try:
            artifact = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate = artifact.get("candidate", artifact)
            evaluation_5d = candidate_path.parent / "train_evaluation_5d/report.json"
            evaluation_legacy = candidate_path.parent / "train_evaluation/report.json"
            evaluation_path = evaluation_5d if evaluation_5d.exists() else evaluation_legacy
            weekly_path = candidate_path.parent / "weekly_backtest/report.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
            weekly = json.loads(weekly_path.read_text(encoding="utf-8")) if weekly_path.exists() else {}
            primary_bps = weekly.get("primary_cost_bps", 10)
            primary = next((x for x in weekly.get("cost_scenarios", []) if x["cost_bps"] == primary_bps), None)
            train_decision = evaluation.get("preliminary_decision", "not_evaluated")
            core_redundancy = evaluation.get("core25_highest_redundancy") or {}
            final = ("weekly_passed" if weekly.get("training_gates_passed") else
                     "weekly_failed" if weekly else train_decision)
            records.append({
                "trial": candidate_path.parent.name, "factor_name": candidate.get("factor_name"),
                "formula": candidate.get("formula"), "inputs": candidate.get("inputs", []),
                "category": category_of(candidate.get("inputs", [])),
                "train_decision": train_decision, "final_outcome": final,
                "label_column": evaluation.get("label_column", "label_ret_20_legacy"),
                "rank_ic": evaluation.get("performance", {}).get("rank_ic"),
                "incremental_rank_ic": core_redundancy.get("partial_rank_ic"),
                "max_core_abs_corr": core_redundancy.get("daily_abs_rank_corr_mean"),
                "weekly_10bps": ({k: primary.get(k) for k in
                    ("annual_excess_return", "sharpe", "max_drawdown", "average_weekly_one_way_turnover")}
                    if primary else None),
            })
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    field_counts = Counter(field for record in records for field in record["inputs"])
    # 方向地图:每个经济大类的试验数 / 通过数 / 最佳|rank_ic| / 平均换手。
    category_map = {}
    for record in records:
        cat = record["category"]
        bucket = category_map.setdefault(cat, {"trials": 0, "passed": 0,
                                               "best_abs_rank_ic": 0.0, "turnovers": []})
        bucket["trials"] += 1
        bucket["passed"] += int(record["final_outcome"] == "weekly_passed")
        if record["rank_ic"] is not None:
            bucket["best_abs_rank_ic"] = max(bucket["best_abs_rank_ic"], abs(record["rank_ic"]))
        if record["weekly_10bps"] and record["weekly_10bps"].get("average_weekly_one_way_turnover") is not None:
            bucket["turnovers"].append(record["weekly_10bps"]["average_weekly_one_way_turnover"])
    category_direction = {}
    for cat, bucket in category_map.items():
        category_direction[cat] = {
            "label": CATEGORY_LABEL.get(cat, cat), "trials": bucket["trials"],
            "passed": bucket["passed"], "best_abs_rank_ic": bucket["best_abs_rank_ic"],
            "mean_weekly_turnover": round(sum(bucket["turnovers"]) / len(bucket["turnovers"]), 3) if bucket["turnovers"] else None,
            "status": ("open" if bucket["passed"] > 0 else
                       "closed_sterile" if bucket["trials"] >= 3 and bucket["best_abs_rank_ic"] < 0.02 else
                       "probing"),
        }
    memory = {
        "version": 3, "active_label": "label_ret_5", "trial_count": len(records),
        "outcome_counts": dict(Counter(record["final_outcome"] for record in records)),
        "field_usage": dict(field_counts.most_common()),
        "category_direction": category_direction,
        "exact_formulas_do_not_repeat": sorted({r["formula"] for r in records if r["formula"]}),
        "recent_experience": records[-30:],
        "generation_rules": [
            "Do not repeat exact formulas or trivial algebraic equivalents.",
            "Use weekly 10bps failure as evidence against turnover-heavy concepts.",
            "Generate a new economic logic before tuning its window.",
            "Prefer underexplored fields and concepts unless feedback supports refinement.",
            "Only compare admission metrics from label_ret_5; older horizons are qualitative memory only.",
            "Respect category_direction.status: closed_sterile categories (>=3 trials, best|rank_ic|<0.02) are forbidden new proposals.",
            "Winning QM candidates are slow, negative-direction risk concepts with low weekly turnover (<=0.27).",
        ],
    }
    OUTPUT.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ready", "trials": len(records), "output": str(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
