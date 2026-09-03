"""对已审计测试集进行实现错误纠正后的复算，不覆盖原始测试结果。"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.config import PortfolioConfig
from backtest.engine import run_portfolio_backtest
from backtest.run_backtest import _load_inputs
from config.settings import BACKTEST_REPORTS_DIR, SELECTED_MODEL_PATH, TEST_EVALUATION_LOCK


CORRECTION_ID = "open_limit_v2"
REASON = "开盘成交不再使用当日全天 pct_chg 涨跌停状态；改用真实昨收与开盘价。"


def _hash(config: PortfolioConfig, model: str, label: str) -> str:
    payload = json.dumps({"model": model, "label": label, **config.to_dict()},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    if not TEST_EVALUATION_LOCK.exists():
        raise RuntimeError("缺少原始测试审计锁；纠错复算只允许在原测试已完成后执行")
    original_audit = json.loads(TEST_EVALUATION_LOCK.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED_MODEL_PATH.read_text(encoding="utf-8"))
    model, label = selected["model"], selected["label"]
    config = PortfolioConfig()
    config_hash = _hash(config, model, label)
    if config_hash == original_audit.get("config_hash"):
        raise RuntimeError("新旧配置哈希相同，拒绝无原因重复查看测试集")

    out = BACKTEST_REPORTS_DIR / "test_corrections" / CORRECTION_ID / model / label
    audit_path = BACKTEST_REPORTS_DIR / "test_corrections" / CORRECTION_ID / "audit.json"
    if audit_path.exists():
        raise RuntimeError(f"该纠错版本已经评估并锁定: {audit_path}")

    pred, market, benchmark = _load_inputs(model, label, "test")
    result = run_portfolio_backtest(pred, market, benchmark, config)
    out.mkdir(parents=True, exist_ok=True)
    result.nav.to_parquet(out / "nav.parquet", index=False)
    result.trades.to_parquet(out / "trades.parquet", index=False)
    result.holdings.to_parquet(out / "holdings.parquet", index=False)
    summary = {
        "segment": "test_correction", "correction_id": CORRECTION_ID,
        "correction_reason": REASON, "model": model, "label": label,
        "config": config.to_dict(), "config_hash": config_hash,
        "original_config_hash": original_audit.get("config_hash"),
        "metrics": result.metrics, "parameters_retuned": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "correction_id": CORRECTION_ID, "reason": REASON,
        "original_audit": str(TEST_EVALUATION_LOCK),
        "original_config_hash": original_audit.get("config_hash"),
        "corrected_config_hash": config_hash, "summary": str(out / "summary.json"),
        "parameters_retuned": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
