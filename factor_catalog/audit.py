from __future__ import annotations
import json
from collections import Counter
from config.settings import ROOT
from .build import validate_catalog


def main() -> int:
    path = ROOT / "config" / "factor_catalog.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    validate_catalog(catalog)
    layer2 = [item for item in catalog["factors"] if item["layer"] == 2 and item["in_current_270"]]
    result = {
        "physical_layer2": len(layer2), "identities": len(catalog["factors"]),
        "internal_atomic": sum(item["source"] == "internal" for item in layer2),
        "alpha158": sum(item["source_bundle"] == "qlib_alpha158" for item in layer2),
        "layer3_composite": sum(item["layer"] == 3 for item in catalog["factors"]),
        "active_layer2": sum(item.get("status") != "deprecated" for item in layer2),
        "deprecated_aliases": sum(item.get("status") == "deprecated" for item in layer2),
        "formula_fixes_v2": sum(item.get("formula_revision") == 2 for item in layer2),
        "categories": dict(Counter(item["primary_category"] for item in layer2)),
        "missing_inputs": sum(not item.get("inputs") for item in layer2),
        "errors": [],
    }
    if result["physical_layer2"] != 270: result["errors"].append("第2层物理因子不是270个")
    if result["alpha158"] != 158: result["errors"].append("Alpha158数量异常")
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
