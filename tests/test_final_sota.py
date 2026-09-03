import unittest

from factor_catalog.sota import build_final_sota, validate_final_sota


class FinalSotaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sota = build_final_sota()

    def test_final_sota_has_58_unique_members(self):
        names = [x["factor_name"] for x in self.sota["factors"]]
        self.assertEqual(len(names), 58)
        self.assertEqual(len(set(names)), 58)

    def test_requested_replacements(self):
        by_name = {x["factor_name"]: x for x in self.sota["factors"]}
        self.assertNotIn("OVERNIGHT_RET5", by_name)
        self.assertNotIn("MAX_DD60", by_name)
        self.assertIn("GAP5", by_name)
        self.assertEqual(by_name["CONSEC_UP5"]["formula_revision"], 2)
        self.assertEqual(by_name["MAX_DD60_V2"]["direction"], "negative")
        self.assertEqual(by_name["MAX_DD60_V2"]["member_status"], "provisional")

    def test_validation(self): validate_final_sota(self.sota)

    def test_alpha158_is_an_immutable_separate_benchmark(self):
        alpha = next(x for x in self.sota["benchmark_sources"] if x["benchmark_id"] == "qlib_alpha158")
        self.assertEqual(alpha["factor_count"], 158)
        self.assertIs(alpha["formula_modified"], False)
        self.assertIs(alpha["automatic_sota_membership"], False)

    def test_quantmind_generates_one_to_five_experimental_factors(self):
        policy = self.sota["quantmind_generation_policy"]
        self.assertEqual((policy["min_factors_per_round"], policy["max_factors_per_round"]), (1, 5))
        self.assertEqual(policy["default_status"], "experimental")
        self.assertIs(policy["direct_sota_admission"], False)
        self.assertIs(policy["may_modify_human_sota"], False)
        self.assertIs(policy["may_modify_alpha158"], False)


if __name__ == "__main__": unittest.main()
