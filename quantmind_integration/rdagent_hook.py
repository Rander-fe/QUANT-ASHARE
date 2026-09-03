"""RD-Agent非侵入式候选校验钩子；仅在安装rdagent的环境内导入。"""
from __future__ import annotations

import json
from pathlib import Path

from quantmind_integration.policy import validate_candidate_batch
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesis2Experiment


class ValidatedQlibFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    """在RD-Agent把LLM响应转换为FactorTask之前执行强制校验。"""

    def prepare_context(self, hypothesis, trace):
        context, enabled = super().prepare_context(hypothesis, trace)
        root = Path(__file__).resolve().parents[1]
        appendix = (root / "config" / "quantmind_candidate_prompt_appendix.md").read_text(
            encoding="utf-8"
        )
        memory_path = root / "reports" / "factor_mining_memory.json"
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            active = [r for r in memory.get("recent_experience", []) if r.get("label_column") == "label_ret_5"]
            memory_context = {
                "active_label": "label_ret_5",
                "exact_formulas_do_not_repeat": memory.get("exact_formulas_do_not_repeat", []),
                "recent_same_horizon_experience": active[-20:],
            }
            appendix += "\n\n结构化因子挖掘记忆（只把同为5日口径的指标用于比较）：\n" + json.dumps(
                memory_context, ensure_ascii=False
            )
        context["experiment_output_format"] = f"{context['experiment_output_format']}\n\n{appendix}"
        return context, enabled

    def convert_response(self, response, hypothesis, trace):
        payload = json.loads(response)
        candidates = []
        for factor_name, spec in payload.items():
            raw_inputs = spec.get("inputs")
            inputs = ([item.strip().lstrip("$") for item in raw_inputs.split(",") if item.strip()]
                      if isinstance(raw_inputs, str) else raw_inputs)
            raw_lookback = spec.get("lookback")
            lookback = int(raw_lookback) if isinstance(raw_lookback, str) and raw_lookback.isdigit() else raw_lookback
            candidates.append({
                "factor_name": factor_name,
                "formula": spec.get("formula", spec.get("formulation")),
                "inputs": inputs,
                "lookback": lookback,
                "availability": spec.get("availability"),
                "direction": spec.get("direction"),
                "economic_rationale": spec.get("economic_rationale", spec.get("description")),
            })
        validate_candidate_batch(candidates)
        # 原转换器读取 formulation；统一为已经验证的安全DSL公式。
        for spec in payload.values():
            spec["formulation"] = spec.get("formula", spec.get("formulation"))
        return super().convert_response(json.dumps(payload, ensure_ascii=False), hypothesis, trace)
