from __future__ import annotations

import email.utils
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timezone
from typing import Optional

from .schema import TextDocument, utc_now

_TAG_RE = re.compile(r"<[^>]+>")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, names: set[str]) -> str:
    for child in element.iter():
        if child is element:
            continue
        if _local_name(child.tag) in names and child.text:
            return child.text.strip()
    return ""


def _clean_html(value: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", value or ""))


def parse_datetime(value: str) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    candidate = value.strip().replace("Z", "+00:00")
    try:
        from datetime import datetime
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def parse_feed(
    payload: bytes, *, source: str, document_type: str, collected_at: Optional[str] = None,
) -> list[TextDocument]:
    prefix = payload.lstrip()[:100].lower()
    if prefix.startswith(b"<script") or prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        raise ValueError("Source returned an HTML/anti-bot page instead of RSS/Atom XML")
    root = ET.fromstring(payload)
    items = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
    seen = collected_at or utc_now()
    documents: list[TextDocument] = []
    for item in items:
        title = _child_text(item, {"title"})
        content = _child_text(item, {"content", "encoded", "summary", "description"})
        link = _child_text(item, {"link"})
        if not link:
            for child in item.iter():
                if _local_name(child.tag) == "link" and child.attrib.get("href"):
                    link = child.attrib["href"]
                    break
        external_id = _child_text(item, {"guid", "id"}) or link or None
        published = _child_text(item, {"pubdate", "published", "updated", "date"})
        updated = _child_text(item, {"updated"})
        try:
            documents.append(TextDocument(
                document_type=document_type,
                source=source,
                title=_clean_html(title),
                content=_clean_html(content),
                source_url=link,
                published_at=parse_datetime(published),
                updated_at=parse_datetime(updated),
                first_seen_at=seen,
                collected_at=seen,
                external_id=external_id,
            ))
        except ValueError:
            continue
    return documents


def fetch_feed(
    url: str, *, source: str, document_type: str, timeout: int = 20,
) -> list[TextDocument]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "QUANT-ASHARE-TextCollector/1.0 (+research; contact=local)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(10 * 1024 * 1024)
    return parse_feed(payload, source=source, document_type=document_type)
