from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import ROOT

DEFAULT_CATALOG = ROOT / "config" / "raw_data_catalog.json"


class CatalogError(ValueError):
    """原始数据目录契约不合法。"""


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_catalog(payload)
    return payload


def validate_catalog(payload: dict[str, Any]) -> None:
    if payload.get("layer") != 0:
        raise CatalogError("raw data catalog 的 layer 必须为 0")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise CatalogError("datasets 必须是非空列表")
    ids: set[str] = set()
    forbidden = {"alpha158", "factors", "features", "labels"}
    for dataset in datasets:
        dataset_id = dataset.get("id")
        if not dataset_id or dataset_id in ids:
            raise CatalogError(f"数据集ID缺失或重复: {dataset_id!r}")
        ids.add(dataset_id)
        paths = dataset.get("paths")
        if not isinstance(paths, list) or not paths:
            raise CatalogError(f"{dataset_id}: paths 必须是非空列表")
        for pattern in paths:
            stem = Path(pattern).stem.lower()
            if stem in forbidden:
                raise CatalogError(f"{dataset_id}: 第0层禁止登记派生因子文件 {pattern}")
        if not dataset.get("availability"):
            raise CatalogError(f"{dataset_id}: 缺少 availability 时间可用性说明")
