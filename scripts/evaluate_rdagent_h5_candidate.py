"""Evaluate an RD-Agent HDF5 factor against the five-day close-to-close label."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2020-09-01")
    args = parser.parse_args()

    factor = pd.read_hdf(args.factor).reset_index()
    score = next(column for column in factor.columns if column not in {"datetime", "instrument"})
    # RD-Agent's official generator writes a Fixed-format HDF5 store, which
    # does not support column projection during read.
    market = pd.read_hdf(args.market)[["$close"]].reset_index()
    market = market.sort_values(["instrument", "datetime"])
    market["label_ret_5"] = market.groupby("instrument")["$close"].shift(-5) / market["$close"] - 1
    data = factor.merge(market[["datetime", "instrument", "label_ret_5"]],
                        on=["datetime", "instrument"], how="left", validate="one_to_one")
    data = data[(data["datetime"] >= args.start) & (data["datetime"] <= args.end)]
    finite = np.isfinite(data[score]) & np.isfinite(data["label_ret_5"])
    valid = data.loc[finite, ["datetime", score, "label_ret_5"]]

    def daily(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "ic": group[score].corr(group["label_ret_5"], method="pearson"),
            "rank_ic": group[score].corr(group["label_ret_5"], method="spearman"),
            "n": len(group),
        })

    by_day = valid.groupby("datetime", sort=True).apply(daily, include_groups=False).dropna()
    by_year = by_day.assign(year=by_day.index.year).groupby("year").agg(
        ic=("ic", "mean"), rank_ic=("rank_ic", "mean"), days=("rank_ic", "size")
    )
    report = {
        "status": "experimental_only",
        "formal_sota_write": False,
        "factor": score,
        "label": "label_ret_5",
        "period": [args.start, args.end],
        "rows": int(len(data)),
        "valid_pairs": int(len(valid)),
        "missing_or_nonfinite_rate": float(1 - finite.mean()) if len(data) else None,
        "ic": float(by_day["ic"].mean()),
        "rank_ic": float(by_day["rank_ic"].mean()),
        "rank_ic_ir": float(by_day["rank_ic"].mean() / by_day["rank_ic"].std()) if len(by_day) > 1 else None,
        "positive_rank_ic_day_rate": float((by_day["rank_ic"] > 0).mean()),
        "yearly": {str(year): {k: float(v) if k != "days" else int(v)
                               for k, v in row.items()}
                   for year, row in by_year.to_dict("index").items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
