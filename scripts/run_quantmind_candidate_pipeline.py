"""One-command train-only pipeline for any generated QUANTMIND candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="QUANTMIND候选因子训练期全流程")
    parser.add_argument("candidate", type=Path, help="DeepSeek生成的candidate.json")
    parser.add_argument("--skip-weekly", action="store_true")
    args = parser.parse_args()
    candidate_path = args.candidate.resolve()
    output = candidate_path.parent / "train_evaluation"
    run_checked([sys.executable, str(ROOT / "scripts/evaluate_quantmind_candidate.py"),
                 str(candidate_path), "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    decision = report["preliminary_decision"]
    weekly_status = "not_eligible"
    if decision == "eligible_for_weekly_backtest" and not args.skip_weekly:
        weekly = candidate_path.parent / "weekly_backtest"
        run_checked([
            sys.executable, str(ROOT / "scripts/run_quantmind_weekly_candidate.py"),
            "--signals", str(output / "factor_values.parquet"),
            "--score-col", report["score_column"],
            "--direction", report["learned_direction_train_only"],
            "--output-dir", str(weekly),
        ])
        weekly_status = "completed"
    elif decision == "eligible_for_weekly_backtest":
        weekly_status = "skipped_by_flag"
    manifest = {
        "pipeline_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(candidate_path), "train_evaluation": str(output / "report.json"),
        "preliminary_decision": decision, "weekly_backtest": weekly_status,
        "validation_rows_read": 0, "test_rows_read": 0,
    }
    (candidate_path.parent / "pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
