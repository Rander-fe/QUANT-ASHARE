from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import ROOT
from .catalog import DEFAULT_CATALOG, load_catalog


DEFAULT_OUTPUT = ROOT / "data" / "catalog" / "raw_data_inventory.json"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _parquet_inventory(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "fields": [
            {"name": field.name, "physical_type": str(field.type)}
            for field in parquet.schema_arrow
        ],
    }


def _json_inventory(paths: list[Path]) -> dict[str, Any]:
    fields: list[str] = []
    if paths:
        try:
            sample = json.loads(paths[0].read_text(encoding="utf-8"))
            fields = sorted(sample) if isinstance(sample, dict) else []
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "sample_path": _relative(paths[0]) if paths else None,
        "sample_fields": fields,
    }


def _sqlite_inventory(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = []
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        for (name,) in connection.execute(query):
            columns = [
                {"name": row[1], "physical_type": row[2], "not_null": bool(row[3]), "primary_key": bool(row[5])}
                for row in connection.execute(f'PRAGMA table_info("{name}")')
            ]
            rows = connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            tables.append({"name": name, "rows": rows, "fields": columns})
        return {"path": _relative(path), "bytes": path.stat().st_size, "tables": tables}
    finally:
        connection.close()


def build_inventory(catalog_path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    datasets = []
    errors: list[str] = []
    warnings: list[str] = []
    for contract in catalog["datasets"]:
        matches: list[Path] = []
        for pattern in contract["paths"]:
            matches.extend(sorted(ROOT.glob(pattern)))
        matches = list(dict.fromkeys(path for path in matches if path.is_file()))
        entry: dict[str, Any] = {
            "id": contract["id"],
            "category": contract["category"],
            "format": contract["format"],
            "availability": contract["availability"],
            "research_ready": contract.get("research_ready", True),
            "usage_policy": contract.get("usage_policy"),
            "known_risks": contract.get("known_risks", []),
            "matched_files": len(matches),
        }
        if not matches:
            errors.append(f"{contract['id']}: 未匹配到文件")
            entry["status"] = "missing"
            datasets.append(entry)
            continue
        if contract["format"] == "parquet":
            physical = [_parquet_inventory(path) for path in matches]
            actual = {field["name"] for item in physical for field in item["fields"]}
            missing = sorted(set(contract.get("required_fields", [])) - actual)
            if missing:
                errors.append(f"{contract['id']}: 缺少必需字段 {missing}")
            entry.update({"status": "invalid" if missing else "ok", "missing_required_fields": missing, "physical": physical})
        elif contract["format"] == "json":
            info = _json_inventory(matches)
            missing = sorted(set(contract.get("required_fields", [])) - set(info["sample_fields"]))
            if missing:
                warnings.append(f"{contract['id']}: 样例缺少字段 {missing}")
            entry.update({"status": "warning" if missing else "ok", "missing_sample_fields": missing, "physical": info})
        elif contract["format"] == "sqlite":
            physical = [_sqlite_inventory(path) for path in matches]
            actual_tables = {table["name"] for item in physical for table in item["tables"]}
            missing = sorted(set(contract.get("required_tables", [])) - actual_tables)
            if missing:
                errors.append(f"{contract['id']}: 缺少必需表 {missing}")
            entry.update({"status": "invalid" if missing else "ok", "missing_required_tables": missing, "physical": physical})
        datasets.append(entry)
    return {
        "catalog_version": catalog["catalog_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_path": _relative(Path(catalog_path)),
        "summary": {
            "datasets": len(datasets),
            "ok": sum(item["status"] == "ok" for item in datasets),
            "warnings": len(warnings),
            "errors": len(errors),
        },
        "warnings": warnings,
        "errors": errors,
        "datasets": datasets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="审计第0层原始数据目录")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inventory = build_inventory(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory["summary"], ensure_ascii=False))
    for warning in inventory["warnings"]:
        print(f"[WARN] {warning}")
    for error in inventory["errors"]:
        print(f"[ERROR] {error}")
    print(f"[OK] inventory: {args.output}")
    return 1 if inventory["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
