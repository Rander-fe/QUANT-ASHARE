"""串行完成简单基准、Ridge、MLP训练并生成统一验证集比较。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(args: list[str]) -> None:
    command = [sys.executable, *args]
    print(f"[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    label = "label_ret_5"
    run(["models/baselines/train.py", "--label", label])
    run(["models/ridge/train.py", "--label", label])
    run(["models/mlp/train.py", "--label", label])
    run(["analysis/compare_models.py", "--label", label, "--segment", "valid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
