"""Build reusable PIT industry atoms from the existing daily panel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import BASIC_EXTRA_PATH
from quantmind_pipeline.industry_atoms import attach_industry_intervals, build_industry_atoms


def main() -> int:
    intervals_path = ROOT / "data/processed/industry/sw_l1_membership_intervals.parquet"
    if not intervals_path.exists():
        raise RuntimeError("请先运行scripts/fetch_industry_history.py")
    daily = pd.read_parquet(BASIC_EXTRA_PATH, columns=["symbol", "date", "adjclose"])
    daily = daily.rename(columns={"adjclose": "close"})
    intervals = pd.read_parquet(intervals_path)
    panel = attach_industry_intervals(daily, intervals)
    atoms = build_industry_atoms(panel)
    output = ROOT / "data/processed/industry/industry_atoms_daily.parquet"
    keep = ["symbol", "date", "industry_code", "industry_name", "industry_ret_1d_loo",
            "market_ret_1d", "IND_REL_RET_20", "IND_RESID_RET_1D", "IND_RESID_MOM_20"]
    atoms[keep].to_parquet(output, index=False)
    report = {
        "status": "ready", "path": str(output.relative_to(ROOT)), "rows": len(atoms),
        "date_range": [str(atoms.date.min().date()), str(atoms.date.max().date())],
        "industry_coverage": float(atoms.industry_code.notna().mean()),
        "atoms": {
            name: {"missing_rate": float(atoms[name].isna().mean()), "infinity_count": int(
                np.isinf(pd.to_numeric(atoms[name], errors="coerce").to_numpy(float)).sum())}
            for name in ["IND_REL_RET_20", "IND_RESID_RET_1D", "IND_RESID_MOM_20"]
        },
        "selection_data_used": False, "test_data_used_for_selection": False,
    }
    report_path = ROOT / "reports/industry_atom_build.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
