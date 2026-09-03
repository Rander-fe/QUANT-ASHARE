from __future__ import annotations

import json
from config.settings import ROOT


def text_factor_readiness() -> dict:
    catalog = json.loads((ROOT / "config" / "raw_data_catalog.json").read_text(encoding="utf-8"))
    datasets = {item["id"]: item for item in catalog["datasets"]}
    policy = datasets.get("text.policy.raw")
    return {
        "policy_title": {"available": policy is not None, "status": "metadata_only" if policy else "missing"},
        "policy_content": {"available": bool(policy and policy.get("research_ready") is True),
                           "status": "blocked_empty_content"},
        "news": {"available": False, "status": "source_not_connected"},
        "announcement": {"available": False, "status": "source_not_connected"},
    }


def assert_text_source_ready(source: str) -> None:
    item = text_factor_readiness().get(source)
    if item is None or not item["available"]:
        raise ValueError(f"文本来源不可用于因子生成: {source} ({item['status'] if item else 'unknown'})")
