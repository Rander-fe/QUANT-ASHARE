from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.settings import ROOT

MANIFEST = ROOT / "config" / "human_core_25_v1.json"


def load_human_final_sota(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest["collection_status"] != "official_active_human_sota"
            or manifest["factor_count"] != 25
            or not manifest.get("active_for_quantmind_seed")):
        raise ValueError("人工核心25清单状态或数量异常")
    by_file: dict[str, list[dict]] = {}
    for item in manifest["factors"]:
        by_file.setdefault(item["data_file"], []).append(item)
    filters = []
    if start: filters.append(("date", ">=", pd.Timestamp(start)))
    if end: filters.append(("date", "<=", pd.Timestamp(end)))
    combined = None
    for relative, items in by_file.items():
        path = (ROOT / relative).resolve()
        if ROOT.resolve() not in path.parents: raise ValueError("SOTA数据文件越出项目目录")
        columns = ["symbol", "date"] + sorted({item["data_column"] for item in items})
        frame = pd.read_parquet(path, columns=columns, filters=filters or None)
        frame["date"] = pd.to_datetime(frame["date"])
        if frame.duplicated(["symbol", "date"]).any(): raise ValueError(f"{relative}存在重复主键")
        rename = {item["data_column"]: item["factor_name"] for item in items}
        frame = frame.rename(columns=rename)
        combined = frame if combined is None else combined.merge(
            frame, on=["symbol", "date"], how="inner", validate="one_to_one"
        )
    expected = [item["factor_name"] for item in manifest["factors"]]
    if combined is None or set(combined.columns) != {"symbol", "date", *expected}:
        raise ValueError("SOTA多文件合并后字段不完整")
    return combined[["symbol", "date"] + expected]
