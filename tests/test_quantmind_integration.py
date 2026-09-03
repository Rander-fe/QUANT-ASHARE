import unittest
import json
from pathlib import Path

from quantmind_integration.policy import audit_integration, assert_selection_period, validate_candidate_batch
from quantmind_integration.sota_loader import load_human_final_sota
from quantmind_integration.text_readiness import assert_text_source_ready, text_factor_readiness


def candidate(formula="TS_MEAN($close, 20)", inputs=None):
    return {"factor_name": "QM_TEST", "formula": formula, "inputs": inputs or ["close"],
            "lookback": 20, "availability": "after_close", "direction": "learned",
            "economic_rationale": "测试"}


class QuantmindIntegrationTest(unittest.TestCase):
    def test_all_32_operators_are_mapped(self): self.assertEqual(audit_integration()["errors"], [])

    def test_valid_candidate(self): validate_candidate_batch([candidate()])

    def test_financial_report_operator_contract(self):
        validate_candidate_batch([candidate("FIN_DELTA_REPORT($roe,1)", ["roe"])])
        with self.assertRaises(ValueError):
            validate_candidate_batch([candidate("FIN_LAG_REPORT($close,1)", ["close"])])
        with self.assertRaises(ValueError):
            validate_candidate_batch([candidate("FIN_LAG_REPORT($roe,0)", ["roe"])])

    def test_pit_industry_atoms_are_allowed(self):
        validate_candidate_batch([candidate("$IND_REL_RET_20", ["IND_REL_RET_20"])])
        validate_candidate_batch([candidate("$IND_RESID_MOM_20", ["IND_RESID_MOM_20"])])

    def test_rejects_unknown_operator_and_field(self):
        with self.assertRaises(ValueError): validate_candidate_batch([candidate("MAGIC($close)")])
        with self.assertRaises(ValueError): validate_candidate_batch([candidate("TS_MEAN($industry,20)", ["industry"])])

    def test_rejects_more_than_five_and_mismatched_inputs(self):
        with self.assertRaises(ValueError): validate_candidate_batch([dict(candidate(), factor_name=f"F{i}") for i in range(6)])
        with self.assertRaises(ValueError): validate_candidate_batch([candidate(inputs=["open"])])

    def test_trial_allows_one_to_five_experimental_factors(self):
        root = Path(__file__).resolve().parents[1]
        policy = json.loads((root / "config" / "quantmind_candidate_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["factors_per_round"], {"min": 1, "max": 5})
        self.assertEqual(policy["default_status"], "experimental")

    def test_native_joint_loop_uses_five_day_target_and_active_core25(self):
        root = Path(__file__).resolve().parents[1]
        evolution = json.loads((root / "config" / "quantmind_native_evolution.json").read_text(encoding="utf-8"))
        self.assertEqual(evolution["prediction_target"]["column"], "label_ret_5")
        self.assertEqual(evolution["prediction_target"]["qlib_expression"], "Ref($close, -5)/$close - 1")
        self.assertTrue(evolution["joint_optimization"]["enabled"])
        self.assertEqual(len(evolution["search_islands"]), 4)
        self.assertEqual(evolution["admission_shell"]["active_human_seed"], "config/human_core_25_v1.json")
        templates = list((root / "vendor" / "quantmind" / "rd-agent" / "rdagent" / "scenarios" /
                          "qlib" / "experiment").glob("*_template/*.yaml"))
        relevant = [path for path in templates if "Ref($close" in path.read_text(encoding="utf-8")]
        self.assertTrue(relevant)
        for path in relevant:
            text = path.read_text(encoding="utf-8")
            self.assertIn("Ref($close, -5)/$close - 1", text)
            self.assertNotIn("Ref($close, -2)/Ref($close, -1) - 1", text)

    def test_test_period_cannot_select_sota(self):
        assert_selection_period("2016-01-01", "2023-04-30")
        with self.assertRaises(ValueError): assert_selection_period("2025-01-02", "2026-08-13")

    def test_unknown_bare_name_is_rejected(self):
        with self.assertRaises(ValueError): validate_candidate_batch([candidate("TS_MEAN(foo,20)", [])])

    def test_active_25_factor_multifile_loader(self):
        frame = load_human_final_sota("2023-04-03", "2023-04-03")
        self.assertEqual(len(frame.columns), 27)
        self.assertIn("MAX_DD60_V2", frame.columns)
        self.assertNotIn("MAX_DD60", frame.columns)

    def test_unavailable_text_sources_are_blocked(self):
        readiness = text_factor_readiness()
        self.assertFalse(readiness["policy_content"]["available"])
        self.assertFalse(readiness["news"]["available"])
        self.assertFalse(readiness["announcement"]["available"])
        with self.assertRaises(ValueError): assert_text_source_ready("policy_content")

    def test_rdagent_hook_is_configured_without_vendor_patch(self):
        root = Path(__file__).resolve().parents[1]
        env = (root / "config" / "quantmind_rdagent.env.example").read_text(encoding="utf-8")
        self.assertIn("ValidatedQlibFactorHypothesis2Experiment", env)
        self.assertTrue((root / "quantmind_integration" / "rdagent_hook.py").exists())
        hook = (root / "quantmind_integration" / "rdagent_hook.py").read_text(encoding="utf-8")
        self.assertIn("experiment_output_format", hook)
        self.assertIn("quantmind_candidate_prompt_appendix.md", hook)

    def test_rdagent_entry_smoke_test_and_litellm_constraint_exist(self):
        root = Path(__file__).resolve().parents[1]
        smoke = root / "scripts" / "smoke_test_rdagent_hook.py"
        self.assertTrue(smoke.exists())
        smoke_text = smoke.read_text(encoding="utf-8")
        self.assertIn("invalid_candidate", smoke_text)
        requirements = (root / "vendor" / "quantmind" / "rd-agent" / "requirements.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("litellm>=1.73,<1.98", requirements)

    def test_preflight_covers_all_ten_pending_expressions(self):
        from scripts.preflight_qlib_operator_mapping import EXPRESSIONS
        self.assertEqual(len(EXPRESSIONS), 10)
        self.assertEqual(set(EXPRESSIONS), {"TS_MEDIAN", "TS_SKEW", "TS_KURT", "TS_COV", "TS_MAD",
                                            "TS_DECAY_LINEAR", "SIGN", "SQRT", "WHERE", "CLIP"})

    def test_all_ten_mappings_are_docker_preflight_confirmed(self):
        root = Path(__file__).resolve().parents[1]
        mapping = json.loads((root / "config" / "quantmind_qlib_operator_mapping.json").read_text(encoding="utf-8"))
        report = json.loads((root / "reports" / "qlib_operator_preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(mapping["runtime_preflight"]["status"], "docker_preflight_passed")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["results"]), set(__import__(
            "scripts.preflight_qlib_operator_mapping", fromlist=["EXPRESSIONS"]
        ).EXPRESSIONS))
        for operator, result in report["results"].items():
            self.assertEqual(result["status"], "passed")
            self.assertEqual(mapping["operators"][operator]["verification"], "docker_preflight_passed")
            self.assertNotIn("preflight", mapping["operators"][operator]["mode"])


if __name__ == "__main__": unittest.main()
