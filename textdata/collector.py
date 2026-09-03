from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import DATA_PROCESSED, DATA_RAW

from .rss import fetch_feed
from .json_source import fetch_json_list
from .schema import utc_now
from .storage import DocumentStore
from .tushare_source import fetch_tushare

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "text_sources.json"
RAW_ROOT = DATA_RAW / "text"
STATE_PATH = DATA_PROCESSED / "text" / "collector_state.sqlite3"


@dataclass
class SourceResult:
    name: str
    status: str
    fetched: int = 0
    inserted: int = 0
    error: str = ""


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("text source config must contain a sources list")
    return sources


def collect(config_path: Path, *, start_date: str, end_date: str) -> list[SourceResult]:
    results: list[SourceResult] = []
    with DocumentStore(RAW_ROOT, STATE_PATH) as store:
        for source in load_sources(config_path):
            if not source.get("enabled", False):
                continue
            name = str(source["name"])
            started = utc_now()
            run_id = store.start_run(name, started)
            result = SourceResult(name=name, status="running")
            try:
                kind = source["kind"]
                if kind == "rss":
                    documents = fetch_feed(
                        source["url"], source=name,
                        document_type=source["document_type"],
                        timeout=int(source.get("timeout", 20)),
                    )
                elif kind == "json_list":
                    documents = fetch_json_list(
                        source["url"], source=name,
                        document_type=source["document_type"],
                        field_map=source.get("field_map", {}),
                        timeout=int(source.get("timeout", 20)),
                    )
                elif kind == "tushare":
                    documents = fetch_tushare(
                        source["api_name"], document_type=source["document_type"],
                        start_date=start_date, end_date=end_date,
                    )
                else:
                    raise ValueError(f"Unsupported source kind: {kind}")
                result.fetched, result.inserted = store.put_many(documents)
                result.status = "succeeded"
            except Exception as exc:
                result.status = "failed"
                result.error = f"{type(exc).__name__}: {exc}"
            finally:
                store.finish_run(
                    run_id, utc_now(), result.status, result.fetched,
                    result.inserted, result.error,
                )
            results.append(result)
    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Collect point-in-time text documents")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--list-sources", action="store_true")
    args = parser.parse_args()
    if args.list_sources:
        for source in load_sources(args.config):
            print(f"{source.get('name')}\tenabled={source.get('enabled', False)}\t{source.get('kind')}")
        return 0
    today = __import__("datetime").date.today().strftime("%Y%m%d")
    results = collect(
        args.config,
        start_date=args.start_date or today,
        end_date=args.end_date or today,
    )
    failed = False
    for result in results:
        print(
            f"[{result.status}] {result.name}: fetched={result.fetched}, "
            f"inserted={result.inserted}{' error=' + result.error if result.error else ''}"
        )
        failed |= result.status == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
