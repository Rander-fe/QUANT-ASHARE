from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.protocol import assert_point_in_time_contract, write_experiment_manifest


class PointInTimeContractTests(unittest.TestCase):
    def _valid_frame(self):
        return pd.DataFrame({
            "available_date": ["2024-01-02"],
            "signal_date": ["2024-01-03"],
            "execution_date": ["2024-01-04"],
            "label_end_date": ["2024-02-01"],
        })

    def test_valid_temporal_order(self):
        assert_point_in_time_contract(self._valid_frame())

    def test_future_information_is_rejected(self):
        frame = self._valid_frame()
        frame.loc[0, "available_date"] = "2024-01-05"
        with self.assertRaisesRegex(ValueError, "时间契约违规"):
            assert_point_in_time_contract(frame)

    def test_missing_availability_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            assert_point_in_time_contract(self._valid_frame().drop(columns="available_date"))


class ManifestTests(unittest.TestCase):
    def test_manifest_records_inputs_and_test_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.bin"
            source.write_bytes(b"research-data")
            output = root / "manifest.json"
            write_experiment_manifest(
                output, experiment="unit-test", config={"label": "label_ret_20"},
                inputs=[source], features=["f1", "f2"], test_data_used=False,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["test_data_used"])
            self.assertEqual(payload["feature_count"], 2)
            self.assertTrue(payload["inputs"][0]["exists"])
            self.assertEqual(len(payload["inputs"][0]["edge_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
