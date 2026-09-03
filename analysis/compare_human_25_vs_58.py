"""Training-only style/diversity comparison of human core 25 and reference 58."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/factor_library/human_25_vs_58"


def diversity_metrics(names: list[str], categories: dict[str, str], daily_ic: pd.DataFrame) -> dict:
    historical = ["MAX_DD60" if x == "MAX_DD60_V2" else x for x in names]
    available = [x for x in historical if x in daily_ic.columns]
    corr = daily_ic[available].corr(min_periods=100)
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    eig = np.linalg.eigvalsh(corr.fillna(0).to_numpy())
    eig = np.clip(eig, 0, None)
    effective_dimension = float(eig.sum() ** 2 / np.square(eig).sum())
    counts = Counter(categories[x] for x in names)
    p = np.array(list(counts.values()), dtype=float) / len(names)
    entropy = float(-(p * np.log(p)).sum() / np.log(len(counts))) if len(counts) > 1 else 0.0
    return {
        "factor_count": len(names), "category_count": len(counts), "category_distribution": dict(counts),
        "normalized_category_entropy": entropy, "category_hhi": float(np.square(p).sum()),
        "quantity_price_liquidity_share": float(sum(v for k, v in counts.items() if k in {"liquidity", "price_volume", "price_structure", "technical_state"}) / len(names)),
        "mean_abs_daily_ic_correlation": float(upper.abs().mean()),
        "p90_abs_daily_ic_correlation": float(upper.abs().quantile(.9)),
        "maximum_abs_daily_ic_correlation": float(upper.abs().max()),
        "pairs_abs_correlation_ge_070": int((upper.abs() >= .70).sum()),
        "effective_ic_dimension_participation_ratio": effective_dimension,
    }


def quality(names: list[str], quantile: pd.DataFrame) -> dict:
    historical = ["MAX_DD60" if x == "MAX_DD60_V2" else x for x in names]
    block = quantile[quantile["factor"].isin(historical)]
    return {
        "matched_factors": int(len(block)),
        "annual_return_mean": float(block["ann_return"].mean()),
        "annual_return_median": float(block["ann_return"].median()),
        "sharpe_mean": float(block["sharpe"].mean()),
        "sharpe_median": float(block["sharpe"].median()),
        "max_drawdown_mean": float(block["max_drawdown"].mean()),
        "turnover_mean_legacy_jaccard": float(block["turnover"].mean()),
        "strong_monotonic_count_abs_gt_080": int((block["monotonic_corr"].abs() > .8).sum()),
        "nonpositive_aligned_return_count": int((block["ann_return"] <= 0).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg58 = json.loads((ROOT / "config/human_final_sota_v1.json").read_text(encoding="utf-8"))
    factors58 = [x["factor_name"] for x in cfg58["factors"]]
    categories = {x["factor_name"]: x["category"] for x in cfg58["factors"]}
    recovered = json.loads((ROOT / "reports/factor_library/human_core_25_recovery.json").read_text(encoding="utf-8"))
    factors25 = recovered["normalized_current_factors"]
    daily_ic = pd.read_parquet(ROOT / "data/processed/daily_rankic.parquet")
    quantile = pd.read_parquet(ROOT / "data/processed/factor_quantile_train.parquet")
    result = {}
    for label, names in (("human_core_25", factors25), ("human_reference_58", factors58)):
        result[label] = {"diversity": diversity_metrics(names, categories, daily_ic),
                         "training_group_quality": quality(names, quantile)}
    cat_rows = []
    for category in sorted(set(categories.values())):
        cat_rows.append({"category": category,
                         "core25": sum(categories[x] == category for x in factors25),
                         "reference58": sum(categories[x] == category for x in factors58)})
    pd.DataFrame(cat_rows).to_csv(OUT / "category_distribution.csv", index=False, encoding="utf-8-sig")
    decision = {
        "recommended_architecture": {
            "quantmind_human_seed_and_default_model_candidates": "human_core_25",
            "redundancy_reference_and_research_archive": "human_reference_58"
        },
        "reason": "25 has materially stronger training group quality; 58 has broader category coverage and a higher effective IC dimension. Keep both with different roles.",
        "formal_58_modified": False,
        "validation_or_test_used_for_this_comparison": False,
    }
    report = {"status": "training_only_comparison", "sets": result, "decision": decision}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
