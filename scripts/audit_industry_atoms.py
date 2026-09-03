"""Audit an already-built industry atom file without recomputing it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ATOM_PATH = ROOT / "data/processed/industry/industry_atoms_daily.parquet"


def main() -> int:
    metadata = pq.read_metadata(ATOM_PATH)
    columns = ["date", "industry_code", "IND_REL_RET_20", "IND_RESID_RET_1D", "IND_RESID_MOM_20"]
    frame = pd.read_parquet(ATOM_PATH, columns=columns)
    report = {
        "status": "ready", "path": str(ATOM_PATH.relative_to(ROOT)), "rows": metadata.num_rows,
        "date_range": [str(pd.to_datetime(frame.date).min().date()), str(pd.to_datetime(frame.date).max().date())],
        "industry_coverage": float(frame.industry_code.notna().mean()),
        "atoms": {
            name: {"missing_rate": float(frame[name].isna().mean()),
                   "infinity_count": int(np.isinf(pd.to_numeric(frame[name], errors="coerce").to_numpy(float)).sum()),
                   "p01": float(frame[name].quantile(.01)), "median": float(frame[name].median()),
                   "p99": float(frame[name].quantile(.99))}
            for name in ["IND_REL_RET_20", "IND_RESID_RET_1D", "IND_RESID_MOM_20"]
        },
        "selection_data_used": False, "test_data_used_for_selection": False,
    }
    path = ROOT / "reports/industry_atom_build.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
