import json
import tempfile
import unittest
from pathlib import Path

from textdata.rss import parse_feed
from textdata.json_source import parse_json_list
from textdata.schema import TextDocument
from textdata.storage import DocumentStore
from textdata.tushare_source import dataframe_to_documents


RSS_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Example</title><item>
<guid>item-1</guid><title>Policy &amp; market update</title>
<description><![CDATA[<p>Positive policy text.</p>]]></description>
<link>https://example.com/1</link><pubDate>Sun, 31 Aug 2026 10:00:00 +0800</pubDate>
</item></channel></rss>"""


class TextCollectionTest(unittest.TestCase):
    def test_rss_parse_and_point_in_time_fields(self):
        docs = parse_feed(
            RSS_FIXTURE, source="fixture", document_type="policy",
            collected_at="2026-08-31T03:00:00+00:00",
        )
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].title, "Policy & market update")
        self.assertEqual(docs[0].published_at, "2026-08-31T02:00:00+00:00")
        self.assertEqual(docs[0].first_seen_at, "2026-08-31T03:00:00+00:00")

    def test_html_antibot_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "anti-bot"):
            parse_feed(
                b"<script>document.cookie='challenge'</script>",
                source="blocked", document_type="policy",
            )

    def test_json_list_mapping(self):
        payload = json.dumps([{
            "TITLE": "产业政策",
            "URL": "https://example.com/policy/1",
            "DOCRELPUBTIME": "2026-08-31",
        }], ensure_ascii=False).encode("utf-8")
        docs = parse_json_list(
            payload, source="gov", document_type="policy",
            field_map={
                "title": "TITLE", "url": "URL",
                "published_at": "DOCRELPUBTIME",
            },
            collected_at="2026-08-31T12:00:00+00:00",
        )
        self.assertEqual(docs[0].title, "产业政策")
        self.assertEqual(docs[0].published_at, "2026-08-31T00:00:00+00:00")

    def test_content_addressed_store_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = TextDocument(
                document_type="news", source="fixture", title="same",
                first_seen_at="2026-08-31T03:00:00+00:00",
                collected_at="2026-08-31T03:00:00+00:00",
            )
            with DocumentStore(root / "raw", root / "state.sqlite3") as store:
                first, path = store.put(doc)
                second, _ = store.put(doc)
                self.assertTrue(first)
                self.assertFalse(second)
                self.assertTrue(path.exists())
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["content_hash"], doc.content_hash)
                self.assertEqual(store.stats(), {"news": 1})

    def test_tushare_dataframe_mapping(self):
        import pandas as pd
        frame = pd.DataFrame([{
            "ts_code": "600000.SH", "ann_date": "20260831",
            "title": "测试公告", "url": "https://example.com/a.pdf",
        }])
        docs = dataframe_to_documents(
            frame, api_name="anns_d", document_type="announcement"
        )
        self.assertEqual(docs[0].symbols, ["600000.SH"])
        self.assertEqual(docs[0].published_at, "2026-08-31T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
