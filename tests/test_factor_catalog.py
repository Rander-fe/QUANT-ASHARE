import json
import unittest
from pathlib import Path

from factor_catalog.build import build_catalog, validate_catalog


class FactorCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.catalog = build_catalog()

    def test_current_physical_layer_has_270_factors(self):
        current = [f for f in self.catalog["factors"] if f["layer"] == 2 and f["in_current_270"]]
        self.assertEqual(len(current), 270)

    def test_alpha158_is_immutable_and_complete(self):
        alpha = [f for f in self.catalog["factors"] if f["source_bundle"] == "qlib_alpha158"]
        self.assertEqual(len(alpha), 158)
        self.assertTrue(all(f["formula_modified"] is False and f.get("expression") for f in alpha))

    def test_gp_factors_are_not_atomic(self):
        gp = [f for f in self.catalog["factors"] if f["source_bundle"] == "custom" and f["layer"] == 3]
        self.assertEqual(len(gp), 6)
        self.assertTrue(all(not f["in_current_270"] for f in gp))

    def test_all_atomic_factors_have_inputs_and_time_contract(self):
        atomic = [f for f in self.catalog["factors"] if f["layer"] == 2]
        self.assertTrue(all(f["inputs"] and f["availability"] and f["earliest_execution"] for f in atomic))

    def test_known_legacy_issues_are_machine_readable(self):
        by_id = {f["factor_id"]: f for f in self.catalog["factors"]}
        self.assertEqual(by_id["internal.CONSEC_UP3"]["review_status"], "fixed")
        self.assertEqual(by_id["internal.MOM20"]["status"], "deprecated")
        self.assertEqual(by_id["internal.MOM20"]["alias_of"], "internal.REV20")
        self.assertEqual(by_id["internal.MAX_DD20"]["primary_category"], "volatility_risk")
        self.assertEqual(by_id["internal.MAX_DD20"]["direction"], "negative")
        self.assertEqual(by_id["internal.MAX_DD20"]["official_value_status"], "not_replaced")

    def test_validation_passes(self): validate_catalog(self.catalog)
