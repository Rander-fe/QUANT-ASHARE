import json
import tempfile
import unittest
from pathlib import Path

from data_catalog.catalog import CatalogError, load_catalog, validate_catalog


class RawDataCatalogTest(unittest.TestCase):
    def test_project_catalog_is_valid(self):
        catalog = load_catalog()
        self.assertEqual(catalog["layer"], 0)
        self.assertGreaterEqual(len(catalog["datasets"]), 9)
        self.assertTrue(all(item.get("availability") for item in catalog["datasets"]))

    def test_duplicate_dataset_id_is_rejected(self):
        dataset = {"id": "same", "paths": ["data/raw/x.parquet"], "availability": "T+1"}
        with self.assertRaises(CatalogError):
            validate_catalog({"layer": 0, "datasets": [dataset, dataset]})

    def test_factor_outputs_are_rejected_from_layer_zero(self):
        payload = {
            "layer": 0,
            "datasets": [{"id": "bad", "paths": ["data/processed/factors.parquet"], "availability": "T+1"}],
        }
        with self.assertRaises(CatalogError):
            validate_catalog(payload)

    def test_operator_catalog_has_unique_ids(self):
        path = Path(__file__).resolve().parents[1] / "config" / "operator_catalog.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = [item["id"] for item in payload["operators"]]
        self.assertEqual(payload["layer"], 1)
        self.assertEqual(len(ids), len(set(ids)))
