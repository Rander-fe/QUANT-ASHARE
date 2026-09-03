from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .schema import TextDocument, normalize_text, utc_now
from config.settings import get_tushare_token


def _first(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if str(value).strip():
            return value
    return None


def _date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def dataframe_to_documents(frame, *, api_name: str, document_type: str) -> list[TextDocument]:
    seen = utc_now()
    documents: list[TextDocument] = []
    for row in frame.to_dict("records"):
        title = _first(row, ("title", "ann_title", "name"))
        content = _first(row, ("content", "summary", "abstract")) or ""
        url = _first(row, ("url", "pdf_url", "source_url")) or ""
        published = _first(row, ("pub_time", "publish_time", "ann_date", "pub_date", "date"))
        symbol = _first(row, ("ts_code", "symbol"))
        external = _first(row, ("id", "ann_id", "doc_id")) or url
        if not title and not content:
            continue
        documents.append(TextDocument(
            document_type=document_type,
            source=f"tushare:{api_name}",
            title=normalize_text(title),
            content=normalize_text(content),
            source_url=str(url),
            published_at=_date(published),
            first_seen_at=seen,
            collected_at=seen,
            external_id=str(external) if external else None,
            symbols=[str(symbol)] if symbol else [],
            metadata={"provider_fields": sorted(row.keys())},
        ))
    return documents


def fetch_tushare(
    api_name: str, *, document_type: str, start_date: str, end_date: str,
) -> list[TextDocument]:
    token = os.getenv("TUSHARE_TOKEN", "").strip() or get_tushare_token()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required for Tushare text sources")
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("Install tushare to use Tushare text sources") from exc
    pro = ts.pro_api(token)
    frame = pro.query(api_name, start_date=start_date, end_date=end_date)
    return dataframe_to_documents(frame, api_name=api_name, document_type=document_type)
