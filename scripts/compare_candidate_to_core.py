"""Same-period, same-label predictive comparison for a candidate and one human core factor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_quantmind_candidate import daily_ic, neutralize_day, winsor_z
from scripts.evaluate_quantmind_candidate import LABEL


def summarize(daily: pd.DataFrame) -> dict:
    rank = daily["rank_ic"]
    return {"ic": float(daily.ic.mean()), "rank_ic": float(rank.mean()),
            "rank_ic_ir": float(rank.mean()/rank.std()), "days": len(daily)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_values", type=Path)
    parser.add_argument("--candidate-score", required=True)
    parser.add_argument("--core-factor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = pd.read_parquet(args.candidate_values)
    candidate["date"] = pd.to_datetime(candidate["date"])
    core_cfg = json.loads((ROOT / "config/human_core_25_v1.json").read_text(encoding="utf-8"))
    member = next(x for x in core_cfg["factors"] if x["factor_name"] == args.core_factor)
    core = pd.read_parquet(ROOT / member["data_file"], columns=["symbol", "date", member["data_column"]])
    core["date"] = pd.to_datetime(core["date"])
    start, end = candidate.date.min(), candidate.date.max()
    core = core.loc[core.date.between(start, end)].rename(columns={member["data_column"]: "core_raw"})
    size = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                           columns=["symbol", "date", "circ_mv"],
                           filters=[[('date', '>=', start), ('date', '<=', end)]])
    size["date"] = pd.to_datetime(size["date"])
    panel = candidate.merge(core, on=["symbol", "date"], how="inner").merge(
        size, on=["symbol", "date"], how="left", validate="one_to_one")
    panel["candidate_score"] = panel[args.candidate_score]
    panel["candidate_score"] *= -1 if panel["candidate_score"].corr(panel[LABEL], method="spearman") < 0 else 1
    panel["candidate_wz"] = panel.groupby("date", observed=True)["core_raw"].transform(winsor_z)
    panel["core_score"] = np.nan
    for _, idx in panel.groupby("date", sort=False, observed=True).groups.items():
        panel.loc[idx, "core_score"] = neutralize_day(panel.loc[idx]).loc[idx]
    panel["core_score"] *= -1 if panel["core_score"].corr(panel[LABEL], method="spearman") < 0 else 1
    c_daily, h_daily = daily_ic(panel, "candidate_score"), daily_ic(panel, "core_score")
    c_daily["year"], h_daily["year"] = c_daily.date.dt.year, h_daily.date.dt.year
    years = pd.DataFrame({"candidate_rank_ic": c_daily.groupby("year").rank_ic.mean(),
                          "core_rank_ic": h_daily.groupby("year").rank_ic.mean()}).dropna()
    overall_c, overall_h = summarize(c_daily), summarize(h_daily)
    better_years = int((years.candidate_rank_ic > years.core_rank_ic).sum())
    report = {"period": [str(start.date()), str(end.date())], "candidate": overall_c,
              "core_factor": args.core_factor, "core": overall_h,
              "candidate_better_years": better_years, "comparable_years": len(years),
              "training_replacement_signal": bool(overall_c["rank_ic"] > overall_h["rank_ic"] and better_years > len(years)/2),
              "replacement_executed": False,
              "next_requirement": "weekly costs and one validation-period out-of-sample comparison"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    years.to_csv(args.output.with_suffix(".years.csv"), encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
