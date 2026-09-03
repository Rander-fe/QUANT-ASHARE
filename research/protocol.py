"""机器学习选股研究协议。

统一处理时间可用性契约、输入文件指纹和实验清单，防止同名实验在
不同数据/特征/时间切分下被误认为可直接比较。
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def assert_point_in_time_contract(
    frame: pd.DataFrame,
    *,
    available_col: str = "available_date",
    signal_col: str = "signal_date",
    execution_col: str = "execution_date",
    label_end_col: str | None = "label_end_date",
) -> None:
    """验证信息可用日、信号日、成交日和标签结束日的严格时间顺序。"""
    required = [available_col, signal_col, execution_col]
    if label_end_col is not None:
        required.append(label_end_col)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Point-in-Time 审计缺少字段: {missing}")
    dates = {col: pd.to_datetime(frame[col], errors="coerce") for col in required}
    nulls = [col for col, values in dates.items() if values.isna().any()]
    if nulls:
        raise ValueError(f"Point-in-Time 审计存在无效日期: {nulls}")
    invalid = dates[available_col] > dates[signal_col]
    invalid |= dates[signal_col] >= dates[execution_col]
    if label_end_col is not None:
        invalid |= dates[execution_col] >= dates[label_end_col]
    if invalid.any():
        sample = frame.loc[invalid, required].head(5).to_dict("records")
        raise ValueError(f"发现 {int(invalid.sum())} 条时间契约违规，示例: {sample}")


def file_fingerprint(path: Path) -> dict[str, Any]:
    """生成轻量输入指纹；包含文件元数据以及首尾块内容哈希。"""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        head = stream.read(1024 * 1024)
        digest.update(head)
        if stat.st_size > len(head):
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return {
        "path": str(path.resolve()), "exists": True, "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "edge_sha256": digest.hexdigest(),
    }


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_experiment_manifest(
    output: Path,
    *,
    experiment: str,
    config: dict[str, Any],
    inputs: list[Path],
    features: list[str] | None = None,
    test_data_used: bool = False,
) -> Path:
    """落盘统一实验清单；测试数据使用情况必须显式声明。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    payload = {
        "protocol_version": 1,
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "features": features,
        "feature_count": len(features) if features is not None else None,
        "inputs": [file_fingerprint(path) for path in inputs],
        "test_data_used": bool(test_data_used),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
