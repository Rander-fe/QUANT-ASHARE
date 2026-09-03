"""PIT financial-report adapter used by QUANTMIND candidate evaluation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def attach_report_snapshots(
    frame: pd.DataFrame, fields: list[str], events_path: Path
) -> pd.DataFrame:
    """Attach only the requested financial fields and report-period metadata as-of each date."""
    required = {"symbol", "date"}
    if not required <= set(frame):
        raise KeyError(f"日频数据缺少字段: {sorted(required - set(frame))}")
    columns = ["symbol", "ann_date", "report_period", "report_seq", *fields]
    events = pd.read_parquet(events_path, columns=list(dict.fromkeys(columns)))
    events["symbol"] = events["symbol"].astype(str)
    events["ann_date"] = pd.to_datetime(events["ann_date"], errors="coerce")
    events = events.dropna(subset=["symbol", "ann_date", "report_period"])
    events = events.sort_values(["ann_date", "symbol"], kind="mergesort")

    left = frame.drop(columns=[*fields, "report_period", "report_seq"], errors="ignore").copy()
    left["symbol"] = left["symbol"].astype(str)
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    left["__report_adapter_row"] = range(len(left))
    merged = pd.merge_asof(
        left.sort_values(["date", "symbol"], kind="mergesort"),
        events,
        left_on="date",
        right_on="ann_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("__report_adapter_row").drop(columns="__report_adapter_row").reset_index(drop=True)
