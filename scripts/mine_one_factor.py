"""One-command QUANTMIND factor generation and train-only research pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], capture: bool = False) -> str:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"  # Windows 管道下强制子进程 UTF-8 输出，避免 GBK 解码错误
    result = subprocess.run(command, cwd=ROOT, check=False, text=True, env=env,
                            capture_output=capture, encoding="utf-8")
    if result.returncode:
        detail = (result.stderr or result.stdout or "child process failed").strip()[-2000:]
        raise RuntimeError(f"子流程失败: {detail}")
    return result.stdout if capture else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="生成1个候选并完成训练期预测、查重和10bps周频回测")
    parser.add_argument("--candidate", type=Path, help="复用已有candidate.json；省略时调用DeepSeek生成")
    args = parser.parse_args()
    if args.candidate:
        candidate_path = args.candidate.resolve()
    else:
        run([sys.executable, "-m", "scripts.build_factor_mining_memory"])
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                stdout = run([sys.executable, "-m", "scripts.run_deepseek_factor_trial"], capture=True)
                last_error = None
                break
            except RuntimeError as exc:  # DeepSeek偶发返回异常/超时，重试
                last_error = exc
                print(f"[mine_one_factor] DeepSeek 第{_attempt+1}次失败，重试中: {str(exc)[:300]}", flush=True)
        if last_error is not None:
            raise last_error
        generated = json.loads(stdout.strip().splitlines()[-1])
        candidate_path = Path(generated["output"])
        if not candidate_path.is_absolute():
            candidate_path = ROOT / candidate_path
        # Docker generator may report /workspace; map it back to the shared project root.
        if str(candidate_path).startswith("/workspace"):
            candidate_path = ROOT / candidate_path.relative_to("/workspace")

    # 前置 cheap gate(近2年):缺失率/弱IC快速拦截,不消耗核心查重与完整评估算力。
    quick = candidate_path.parent / "quick_screen"
    if not (quick / "report.json").exists():
        run([sys.executable, "-m", "scripts.evaluate_quantmind_candidate", str(candidate_path),
             "--output-dir", str(quick), "--quick"])
    quick_report = json.loads((quick / "report.json").read_text(encoding="utf-8"))
    quick_decision = quick_report.get("preliminary_decision")
    if quick_decision in ("rejected_excessive_missing_rate", "rejected_weak_train_rank_ic"):
        manifest = {"pipeline_version": 3, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "prediction_label": "label_ret_5", "prediction_horizon_trading_days": 5,
                    "candidate_file": str(candidate_path), "candidate": quick_report["candidate"],
                    "train_decision": quick_decision, "replacement_rule": {"threshold": 0.70, "triggered": False, "executed": False},
                    "weekly_10bps": "not_eligible", "weekly_training_gates_passed": False,
                    "formal_sota_write": False, "validation_rows_read": 0, "test_rows_read": 0,
                    "next_step": "reject or retain experimental"}
        path = candidate_path.parent / "pipeline_manifest_v2.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        run([sys.executable, "-m", "scripts.build_factor_mining_memory"])
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    # Keep legacy 20-day reports immutable.  All new admission decisions use
    # the same five-day close-to-close label as the active core-25 library.
    evaluation = candidate_path.parent / "train_evaluation_5d"
    if not (evaluation / "report.json").exists():
        run([sys.executable, "-m", "scripts.evaluate_quantmind_candidate", str(candidate_path),
             "--output-dir", str(evaluation)])
    report = json.loads((evaluation / "report.json").read_text(encoding="utf-8"))
    redundancy = report["core25_highest_redundancy"]
    replacement = {"threshold": 0.70, "triggered": False, "executed": False}
    if redundancy["daily_abs_rank_corr_mean"] >= replacement["threshold"]:
        comparison = candidate_path.parent / "replacement_comparison.json"
        run([sys.executable, "-m", "scripts.compare_candidate_to_core",
             str(evaluation / "factor_values.parquet"), "--candidate-score", report["score_column"],
             "--core-factor", redundancy["core_factor"], "--output", str(comparison)])
        replacement.update({"triggered": True, "comparison": str(comparison),
                            "note": "训练期胜出仍需周频与验证期通过后才执行正式替换"})

    weekly_status, weekly_passed = "not_eligible", False
    if report["preliminary_decision"] == "eligible_for_weekly_backtest":
        weekly = candidate_path.parent / "weekly_backtest"
        run([sys.executable, "-m", "scripts.run_quantmind_weekly_candidate",
             "--signals", str(evaluation / "factor_values.parquet"),
             "--score-col", report["score_column"], "--direction", report["learned_direction_train_only"],
             "--output-dir", str(weekly)])
        weekly_report = json.loads((weekly / "report.json").read_text(encoding="utf-8"))
        weekly_status, weekly_passed = "completed", weekly_report["training_gates_passed"]
    manifest = {
        "pipeline_version": 3, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_label": "label_ret_5", "prediction_horizon_trading_days": 5,
        "candidate_file": str(candidate_path), "candidate": report["candidate"],
        "train_decision": report["preliminary_decision"], "replacement_rule": replacement,
        "weekly_10bps": weekly_status, "weekly_training_gates_passed": weekly_passed,
        "formal_sota_write": False, "validation_rows_read": 0, "test_rows_read": 0,
        "next_step": "frozen validation-period check" if weekly_passed else "reject or retain experimental",
    }
    path = candidate_path.parent / "pipeline_manifest_v2.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if weekly_passed:
        _register_admitted(report, evaluation, weekly, candidate_path)
    run([sys.executable, "-m", "scripts.build_factor_mining_memory"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def _register_admitted(report: dict, evaluation: Path, weekly: Path, candidate_path: Path) -> None:
    """周频10bps主关卡通过后，把该候选加入准入注册表作为后续冗余参照。

    同步检查:同 factor_name 已存在则仅升级 admitted_stage,不重复;与已注册因子的
    相关性若>=0.7,则由调用方此前判定已排除,此处只做防御性提示。
    """
    registry_path = ROOT / "config/quantmind_admitted_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate = report["candidate"]
    weekly_report = json.loads((weekly / "report.json").read_text(encoding="utf-8"))
    s10 = next(x for x in weekly_report["cost_scenarios"] if x["cost_bps"] == 10)
    member = {
        "factor_name": candidate["factor_name"],
        "raw_factor": candidate["factor_name"],
        "formula": candidate.get("formula"),
        "formula_source": "candidate.json DSL formula",
        "inputs": candidate.get("inputs", []),
        "direction": report["learned_direction_train_only"],
        "category": "qm_deepseek_admitted",
        "signal_file": str((evaluation / "factor_values.parquet").relative_to(ROOT)).replace("\\", "/"),
        "score_column": report["score_column"],
        "train_period": report["period"],
        "weekly_10bps": {
            "annual_excess_return": s10["annual_excess_return"],
            "sharpe": s10["sharpe"],
            "max_drawdown": s10["max_drawdown"],
            "average_weekly_one_way_turnover": s10["average_weekly_one_way_turnover"],
        },
        "admitted_stage": "weekly_training_gates_passed_10bps",
        "admitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "validation_one_shot": "pending",
        "test_access": False,
    }
    for existing in registry["factors"]:
        if existing["factor_name"] == member["factor_name"]:
            existing.update(member)
            break
    else:
        registry["factors"].append(member)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[register_admitted] {member['factor_name']} -> {registry_path}", flush=True)


if __name__ == "__main__": raise SystemExit(main())
