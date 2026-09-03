"""必须在QUANTMIND的Qlib容器中运行：用真实Qlib数据执行待确认表达式。"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

EXPRESSIONS = {
    "TS_MEDIAN": "Med($close, 5)",
    "TS_SKEW": "Skew($close, 5)",
    "TS_KURT": "Kurt($close, 5)",
    "TS_COV": "Cov($close, $volume, 5)",
    "TS_MAD": "Mad($close, 5)",
    "TS_DECAY_LINEAR": "WMA($close, 5)",
    "SIGN": "Sign($close-Ref($close,1))",
    "SQRT": "Power(Abs($close), 0.5)",
    "WHERE": "If(Greater($close,Ref($close,1)),1,0)",
    "CLIP": "If(Greater($close,100),100,If(Less($close,1),1,$close))",
}


def main() -> int:
    try:
        import qlib
        from qlib.data import D
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": f"qlib_import_failed: {exc}"}, ensure_ascii=False))
        return 2
    provider = os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data")
    instrument = os.environ.get("QM_PREFLIGHT_INSTRUMENT", "SH600000")
    start = os.environ.get("QM_PREFLIGHT_START", "2020-01-01")
    end = os.environ.get("QM_PREFLIGHT_END", "2020-03-31")
    qlib.init(provider_uri=provider, region="cn")
    results = {}
    for operator, expression in EXPRESSIONS.items():
        try:
            frame = D.features([instrument], [expression], start_time=start, end_time=end, freq="day")
            finite = int(frame.iloc[:, 0].notna().sum()) if not frame.empty else 0
            results[operator] = {"status": "passed" if finite > 0 else "failed",
                                 "expression": expression, "rows": int(len(frame)), "non_null": finite}
        except Exception as exc:
            results[operator] = {"status": "failed", "expression": expression,
                                 "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "status": "passed" if all(item["status"] == "passed" for item in results.values()) else "failed",
        "executed_at": datetime.now(timezone.utc).isoformat(), "provider_uri": provider,
        "instrument": instrument, "period": [start, end], "results": results,
    }
    output = Path(os.environ.get("QM_PREFLIGHT_OUTPUT", "/workspace/reports/qlib_operator_preflight.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
