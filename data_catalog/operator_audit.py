from __future__ import annotations
import json
from config.settings import ROOT
from factors.operators_v2 import SAFE_OPERATORS

CATALOG = ROOT / "config" / "operator_catalog.json"


def audit() -> dict:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    declared = [item["id"] for item in payload["operators"] if item["status"] == "approved"]
    errors = []
    duplicates = sorted({item for item in declared if declared.count(item) > 1})
    missing = sorted(set(declared) - set(SAFE_OPERATORS))
    extra = sorted(set(SAFE_OPERATORS) - set(declared))
    if duplicates: errors.append(f"重复ID: {duplicates}")
    if missing: errors.append(f"缺少实现: {missing}")
    if extra: errors.append(f"未登记实现: {extra}")
    return {"catalog_version": payload["catalog_version"], "approved": len(declared),
            "legacy": len(payload.get("legacy_mapping", {})),
            "deferred": len(payload.get("deferred_operators", {})), "errors": errors}


def main() -> int:
    result = audit(); print(json.dumps(result, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
