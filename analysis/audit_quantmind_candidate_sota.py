"""Admission audit for the selected QUANTMIND candidate. Never reads the test period."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "reports/quantmind_trials/20260901T064435Z"
OUT = TRIAL / "candidate_sota_audit"
TRAIN = (pd.Timestamp("2016-01-01"), pd.Timestamp("2023-04-30"))
VALID = (pd.Timestamp("2023-05-01"), pd.Timestamp("2025-01-01"))
TEST_START = pd.Timestamp("2025-01-02")
RAW = "QM_DS_LIQUIDITY_AMPLIFICATION_TURNOVER_RETVOL"
SCORE = f"{RAW}_WINSOR_Z_SIZE_NEUTRAL"
EPS, MIN_N = 1e-12, 30


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    ok = a.notna() & b.notna() & (b.abs() > EPS)
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    out.loc[ok] = a.loc[ok] / b.loc[ok]
    return out


def winsor_z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < MIN_N:
        return x * np.nan
    lo, hi = x.quantile([.01, .99])
    y = x.clip(lo, hi)
    sd = y.std(ddof=1)
    return (y - y.mean()) / sd if pd.notna(sd) and sd > EPS else y * np.nan


def neutralize_day(day: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=day.index, dtype="float64")
    ok = day["candidate_wz"].notna() & day["circ_mv"].gt(0)
    if ok.sum() < MIN_N:
        return result
    y = day.loc[ok, "candidate_wz"].to_numpy(float)
    size = np.log(day.loc[ok, "circ_mv"].to_numpy(float))
    design = np.column_stack([np.ones(len(size)), size])
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    sd = residual.std(ddof=1)
    if sd > EPS:
        residual = (residual - residual.mean()) / sd
    result.loc[ok] = residual
    return result


def build_candidate() -> tuple[pd.DataFrame, dict]:
    cols = ["symbol", "date", "close", "volume", "turnover_rate", "circ_mv", "industry",
            "limit_up", "limit_down", "lock_limit_up", "lock_limit_down"]
    frame = pd.read_parquet(
        ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet", columns=cols,
        filters=[[('date', '>=', TRAIN[0]), ('date', '<=', VALID[1])]],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].max() >= TEST_START:
        raise RuntimeError("Test-period boundary violation")
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce").mask(lambda x: x <= 0)
    frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce").mask(lambda x: x <= 0)
    g = frame.groupby("symbol", sort=False, observed=True)
    obs = g.cumcount() + 1
    ret = g["close"].pct_change(fill_method=None)
    retvol = ret.groupby(frame["symbol"], sort=False).rolling(20, min_periods=20).std(ddof=1).reset_index(level=0, drop=True)
    turn5 = g["turnover_rate"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    turn20 = g["turnover_rate"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    frame[RAW] = safe_div(turn5, retvol * safe_div(turn20, turn5))
    missing = frame[RAW].isna()
    reasons = {
        "total_rows": int(len(frame)), "missing_count": int(missing.sum()),
        "missing_rate": float(missing.mean()),
        "insufficient_listing_history_lt_21": int((missing & (obs < 21)).sum()),
        "turnover_input_missing_or_nonpositive": int((missing & (obs >= 21) & frame["turnover_rate"].isna()).sum()),
        "return_volatility_missing_or_zero": int((missing & (obs >= 21) & (retvol.isna() | (retvol.abs() <= EPS))).sum()),
    }
    explained = reasons["insufficient_listing_history_lt_21"] + reasons["turnover_input_missing_or_nonpositive"] + reasons["return_volatility_missing_or_zero"]
    reasons["overlap_or_other"] = int(max(reasons["missing_count"] - explained, 0))
    frame["candidate_wz"] = frame.groupby("date", sort=False, observed=True)[RAW].transform(winsor_z)
    # Assign by the original row index. groupby.apply may concatenate date groups in
    # date order and silently misalign values with the symbol-sorted frame.
    frame[SCORE] = np.nan
    for _, indices in frame.groupby("date", sort=False, observed=True).groups.items():
        frame.loc[indices, SCORE] = neutralize_day(frame.loc[indices]).loc[indices]
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet", columns=["symbol", "date", "label_ret_20"],
                             filters=[[('date', '>=', TRAIN[0]), ('date', '<=', VALID[1])]])
    labels["date"] = pd.to_datetime(labels["date"])
    frame = frame.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")
    return frame, reasons


def ic_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for date, day in frame.groupby("date", sort=True, observed=True):
        x = day[[SCORE, "label_ret_20"]].dropna()
        if len(x) >= MIN_N:
            rows.append({"date": date, "n": len(x), "ic": x[SCORE].corr(x["label_ret_20"]),
                         "rank_ic": x[SCORE].corr(x["label_ret_20"], method="spearman")})
    daily = pd.DataFrame(rows)
    summary = {}
    for col in ("ic", "rank_ic"):
        x = daily[col].dropna()
        mean, std = float(x.mean()), float(x.std())
        # Candidate has negative direction; aligned score is -SCORE.
        summary[col] = {"raw_mean": mean, "aligned_mean": -mean, "aligned_ir": -mean / std,
                        "aligned_correct_rate": float((x < 0).mean()), "days": int(len(x))}
    return daily, summary


def sota_incremental(candidate: pd.DataFrame, config_name: str) -> pd.DataFrame:
    cfg = json.loads((ROOT / f"config/{config_name}").read_text(encoding="utf-8"))
    rows = []
    for member in cfg["factors"]:
        name, source, column = member["factor_name"], ROOT / member["data_file"], member["data_column"]
        factor = pd.read_parquet(source, columns=["symbol", "date", column],
                                 filters=[[('date', '>=', TRAIN[0]), ('date', '<=', TRAIN[1])]])
        factor["date"] = pd.to_datetime(factor["date"])
        merged = candidate[["symbol", "date", SCORE, "label_ret_20"]].merge(
            factor.rename(columns={column: "sota"}), on=["symbol", "date"], how="inner")
        daily_corr, daily_partial = [], []
        for _, day in merged.groupby("date", sort=False, observed=True):
            x = day[[SCORE, "label_ret_20", "sota"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < MIN_N:
                continue
            corr = x.rank().corr(method="pearson")
            r_cl, r_cf, r_lf = corr.loc[SCORE, "label_ret_20"], corr.loc[SCORE, "sota"], corr.loc["label_ret_20", "sota"]
            den = np.sqrt(max(1 - r_cf * r_cf, 0) * max(1 - r_lf * r_lf, 0))
            daily_corr.append(r_cf)
            if den > EPS:
                daily_partial.append((r_cl - r_cf * r_lf) / den)
        rows.append({"sota_factor": name, "days": len(daily_corr),
                     "candidate_sota_rank_corr_mean": float(np.nanmean(daily_corr)),
                     "candidate_sota_abs_rank_corr_mean": float(np.nanmean(np.abs(daily_corr))),
                     "candidate_partial_rank_ic_raw": float(np.nanmean(daily_partial)),
                     "candidate_partial_rank_ic_aligned": -float(np.nanmean(daily_partial))})
        print(f"checked {cfg['sota_id']} {len(rows):02d}/{len(cfg['factors'])}: {name}", flush=True)
    return pd.DataFrame(rows).sort_values("candidate_sota_abs_rank_corr_mean", ascending=False)


def tradability(frame: pd.DataFrame) -> dict:
    results = {}
    for label, period in (("train", TRAIN), ("validation", VALID)):
        part = frame[(frame["date"] >= period[0]) & (frame["date"] <= period[1])].copy()
        daily = []
        previous_long, previous_short = set(), set()
        for date, day in part.groupby("date", sort=True, observed=True):
            day = day.dropna(subset=[SCORE])
            if len(day) < 100:
                continue
            # Negative direction: low factor is long, high factor is short.
            lo, hi = day[SCORE].quantile([.1, .9])
            long = day[day[SCORE] <= lo]
            short = day[day[SCORE] >= hi]
            long_set, short_set = set(long["symbol"]), set(short["symbol"])
            one_way = np.nan
            if previous_long and previous_short:
                one_way = .5 * (1 - len(long_set & previous_long) / max(len(long_set), 1) +
                                 1 - len(short_set & previous_short) / max(len(short_set), 1))
            previous_long, previous_short = long_set, short_set
            daily.append({
                "date": date, "one_way_turnover": one_way,
                "long_limit_up_rate": float(long["limit_up"].fillna(False).mean()),
                "long_lock_up_rate": float(long["lock_limit_up"].fillna(False).mean()),
                "short_limit_down_rate": float(short["limit_down"].fillna(False).mean()),
                "short_lock_down_rate": float(short["lock_limit_down"].fillna(False).mean()),
                "long_suspended_rate": float((long["volume"].fillna(0) <= 0).mean()),
                "short_suspended_rate": float((short["volume"].fillna(0) <= 0).mean()),
            })
        daily_df = pd.DataFrame(daily)
        turnover = float(daily_df["one_way_turnover"].mean())
        results[label] = {c: float(daily_df[c].mean()) for c in daily_df.columns if c != "date"}
        results[label]["annualized_one_way_turnover_252"] = turnover * 252
        results[label]["estimated_annual_cost"] = {
            f"{bps}_bps": turnover * 252 * bps / 10000 for bps in (10, 20, 30)
        }
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, missing = build_candidate()
    if frame["date"].max() >= TEST_START:
        raise RuntimeError("Test data was loaded")
    train_frame = frame[(frame["date"] >= TRAIN[0]) & (frame["date"] <= TRAIN[1])]
    valid_frame = frame[(frame["date"] >= VALID[0]) & (frame["date"] <= VALID[1])]
    train_daily, train_ic = ic_metrics(train_frame)
    valid_daily, valid_ic = ic_metrics(valid_frame)
    train_daily.assign(period="train").to_parquet(OUT / "train_daily_ic.parquet", index=False)
    valid_daily.assign(period="validation").to_parquet(OUT / "validation_daily_ic.parquet", index=False)

    core_redundancy = sota_incremental(train_frame, "human_core_25_v1.json")
    core_redundancy.to_csv(OUT / "core25_correlation_incremental_ic.csv", index=False, encoding="utf-8-sig")
    trading = tradability(frame)
    industry_diag = {}
    for label, part in (("train", train_frame), ("validation", valid_frame)):
        vals = []
        for _, day in part.groupby("date", observed=True):
            x = day[[SCORE, "industry"]].dropna()
            if len(x) < MIN_N:
                continue
            ranks = x[SCORE].rank(pct=True)
            groups = pd.DataFrame({"x": ranks, "industry": x["industry"]}).groupby("industry")["x"]
            total = float(((ranks - ranks.mean()) ** 2).sum())
            vals.append(float((groups.size() * (groups.mean() - ranks.mean()) ** 2).sum()) / total if total > EPS else np.nan)
        industry_diag[label] = {"eta_squared_mean": float(np.nanmean(vals)), "diagnostic_only": True}

    highest_core = core_redundancy.iloc[0].to_dict()
    report = {
        "status": "candidate_sota_admission_audit", "formal_sota_modified": False,
        "candidate": SCORE, "raw_candidate": RAW, "direction": "negative",
        "data_boundary": {"train": [str(TRAIN[0].date()), str(TRAIN[1].date())],
                          "validation_one_shot": [str(VALID[0].date()), str(VALID[1].date())],
                          "test_start": str(TEST_START.date()), "test_rows_read": 0},
        "preprocessing": ["daily 1/99 percentile winsorization", "daily z-score",
                          "OLS neutralization against log(circ_mv)", "residual z-score"],
        "missing_analysis": missing,
        "missing_policy": "No zero imputation. Missing candidate remains NaN; stocks require 21 observations, positive turnover_rate, valid return volatility, and positive circ_mv.",
        "train_ic": train_ic, "validation_ic_one_shot": valid_ic,
        "active_core25": {"count": int(len(core_redundancy)),
                          "highest_abs_rank_correlation": highest_core,
                          "max_abs_corr_threshold": 0.70,
                          "all_individual_results": "core25_correlation_incremental_ic.csv",
                          "role": "primary admission and incremental comparison"},
        "inactive_reference58": {"used": False, "role": "historical archive only"},
        "tradability_and_cost": trading, "industry_exposure": industry_diag,
        "industry_warning": "Current industry is a present-day snapshot backfilled historically; diagnostic only and excluded from admission logic.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    frame[["symbol", "date", RAW, "candidate_wz", SCORE, "label_ret_20", "circ_mv", "industry",
           "volume", "limit_up", "limit_down", "lock_limit_up", "lock_limit_down"]].to_parquet(
        OUT / "candidate_audited_values.parquet", index=False)
    print(json.dumps({"train": train_ic, "validation": valid_ic,
                      "highest_core25_redundancy": highest_core,
                      "missing": missing, "trading": trading}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
