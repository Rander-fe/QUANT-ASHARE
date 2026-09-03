"""Smoke-test the RD-Agent factor-generation entry without calling an LLM."""
from __future__ import annotations

import json
from types import SimpleNamespace

from rdagent.app.qlib_rd_loop.conf import FactorBasePropSetting
from rdagent.core.utils import import_class


HOOK = "quantmind_integration.rdagent_hook.ValidatedQlibFactorHypothesis2Experiment"


def candidate_payload(formula: str = "TS_MEAN($close, 20)") -> dict:
    return {
        "QM_ENTRY_SMOKE": {
            "description": "RD-Agent入口冒烟测试",
            "formula": formula,
            "formulation": formula,
            "variables": {"close": "收盘价"},
            "inputs": ["close"],
            "lookback": 20,
            "availability": "after_close",
            "direction": "learned",
            "economic_rationale": "验证入口，不用于因子评价",
        }
    }


def main() -> int:
    setting = FactorBasePropSetting()
    if setting.hypothesis2experiment != HOOK:
        raise RuntimeError(f"RD-Agent入口未指向校验钩子: {setting.hypothesis2experiment}")
    hook_class = import_class(setting.hypothesis2experiment)
    hook = hook_class()

    trace = SimpleNamespace(hist=[])
    experiment = hook.convert_response(json.dumps(candidate_payload()), None, trace)
    if len(experiment.tasks) != 1 or experiment.tasks[0].factor_name != "QM_ENTRY_SMOKE":
        raise RuntimeError("合法候选没有被上游RD-Agent转换器接收")

    blocked = False
    try:
        hook.convert_response(json.dumps(candidate_payload("MAGIC($close)")), None, trace)
    except ValueError:
        blocked = True
    if not blocked:
        raise RuntimeError("非法候选未被钩子拦截")

    print(json.dumps({
        "status": "passed",
        "configured_entry": setting.hypothesis2experiment,
        "loaded_class": f"{hook_class.__module__}.{hook_class.__name__}",
        "valid_candidate": "accepted",
        "invalid_candidate": "blocked_before_upstream_conversion",
        "llm_called": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
