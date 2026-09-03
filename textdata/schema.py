from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


@dataclass
class TextDocument:
    document_type: str
    source: str
    title: str
    content: str = ""
    source_url: str = ""
    published_at: Optional[str] = None
    first_seen_at: str = field(default_factory=utc_now)
    collected_at: str = field(default_factory=utc_now)
    updated_at: Optional[str] = None
    external_id: Optional[str] = None
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.title = normalize_text(self.title)
        self.content = normalize_text(self.content)
        self.source_url = str(self.source_url or "").strip()
        if self.document_type not in {"news", "policy", "announcement"}:
            raise ValueError(f"Unsupported document_type: {self.document_type}")
        if not self.title and not self.content:
            raise ValueError("A document must contain a title or content")

    @property
    def content_hash(self) -> str:
        payload = f"{self.title}\n{self.content}".encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()

    @property
    def document_id(self) -> str:
        identity = self.external_id or self.source_url or self.content_hash
        payload = f"{self.source}|{identity}".encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["document_id"] = self.document_id
        payload["content_hash"] = self.content_hash
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

