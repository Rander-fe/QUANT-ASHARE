from __future__ import annotations

import json
from typing import Any

import requests

from .schema import TextDocument, normalize_text, utc_now
from .tushare_source import _date


def parse_json_list(
    payload: bytes, *, source: str, document_type: str,
    field_map: dict[str, str], collected_at: str | None = None,
) -> list[TextDocument]:
    data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("JSON source must return a top-level list")
    seen = collected_at or utc_now()
    documents: list[TextDocument] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        title = normalize_text(row.get(field_map.get("title", "title"), ""))
        content = normalize_text(row.get(field_map.get("content", "content"), ""))
        url = str(row.get(field_map.get("url", "url"), "") or "").strip()
        published = row.get(field_map.get("published_at", "published_at"))
        external = row.get(field_map.get("external_id", "external_id")) or url
        if not title and not content:
            continue
        documents.append(TextDocument(
            document_type=document_type,
            source=source,
            title=title,
            content=content,
            source_url=url,
            published_at=_date(published),
            first_seen_at=seen,
            collected_at=seen,
            external_id=str(external) if external else None,
            metadata={"provider_fields": sorted(row.keys())},
        ))
    return documents


def fetch_json_list(
    url: str, *, source: str, document_type: str,
    field_map: dict[str, str], timeout: int = 20,
) -> list[TextDocument]:
    response = requests.get(
        url, timeout=timeout,
        headers={"User-Agent": "QUANT-ASHARE-TextCollector/1.0 (+research; contact=local)"},
    )
    response.raise_for_status()
    payload = response.content
    if len(payload) > 50 * 1024 * 1024:
        raise ValueError("JSON source response exceeds the 50 MiB safety limit")
    return parse_json_list(
        payload, source=source, document_type=document_type, field_map=field_map,
    )
