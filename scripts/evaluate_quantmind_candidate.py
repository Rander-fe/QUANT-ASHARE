"""Generic train-only evaluator for one policy-validated candidate.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantmind_integration.policy import FINANCIAL_FIELDS, assert_selection_period, validate_candidate_batch
from quantmind_pipeline import evaluate_formula
from quantmind_pipeline.financial_reports import attach_report_snapshots


START, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2023-04-30")
MIN_N, EPS = 30, 1e-12
LABEL = "label_ret_5"


def winsor_z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < MIN_N:
        return x * np.nan
    lo, hi = x.quantile([.01, .99])
    y = x.clip(lo, hi)
    sd = y.std(ddof=1)
    return (y - y.mean()) / sd if pd.notna(sd) and sd > EPS else y * np.nan


def neutralize_day(day: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=day.index, dtype="float64")
    ok = day["candidate_wz"].notna() & day["circ_mv"].gt(0)
    if ok.sum() < MIN_N:
        return out
    y = day.loc[ok, "candidate_wz"].to_numpy(float)
    size = np.log(day.loc[ok, "circ_mv"].to_numpy(float))
    x = np.column_stack([np.ones(len(size)), size])
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    sd = residual.std(ddof=1)
    out.loc[ok] = (residual - residual.mean()) / sd if sd > EPS else np.nan
    return out


def load_candidate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", payload)
    validate_candidate_batch([candidate])
    return candidate


def daily_ic(panel: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for date, day in panel.groupby("date", sort=True, observed=True):
        x = day[[score, LABEL]].dropna()
        if len(x) >= MIN_N and x[score].nunique() > 1:
            rows.append({"date": date, "n": len(x), "ic": x[score].corr(x[LABEL]),
                         "rank_ic": x[score].corr(x[LABEL], method="spearman")})
    return pd.DataFrame(rows)


def compare_core25(panel: pd.DataFrame, score: str) -> pd.DataFrame:
    cfg = json.loads((ROOT / "config/human_core_25_v1.json").read_text(encoding="utf-8"))
    rows = []
    for member in cfg["factors"]:
        source, column = ROOT / member["data_file"], member["data_column"]
        old = pd.read_parquet(source, columns=["symbol", "date", column],
                              filters=[[('date', '>=', START), ('date', '<=', END)]])
        old["date"] = pd.to_datetime(old["date"])
        merged = panel[["symbol", "date", score, LABEL]].merge(
            old.rename(columns={column: "core"}), on=["symbol", "date"], how="inner")
        correlations, partials = [], []
        for _, day in merged.groupby("date", sort=False, observed=True):
            x = day[[score, LABEL, "core"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < MIN_N:
                continue
            corr = x.rank().corr()
            r_cl, r_cf, r_lf = corr.loc[score, LABEL], corr.loc[score, "core"], corr.loc[LABEL, "core"]
            den = np.sqrt(max(1-r_cf*r_cf, 0) * max(1-r_lf*r_lf, 0))
            correlations.append(r_cf)
            if den > EPS:
                partials.append((r_cl-r_cf*r_lf)/den)
        rows.append({"core_factor": member["factor_name"], "daily_rank_corr_mean": float(np.nanmean(correlations)),
                     "daily_abs_rank_corr_mean": float(np.nanmean(np.abs(correlations))),
                     "partial_rank_ic": float(np.nanmean(partials))})
        print(f"core25 {len(rows):02d}/25 {member['factor_name']}", flush=True)
    return pd.DataFrame(rows).sort_values("daily_abs_rank_corr_mean", ascending=False)


def compare_admitted_registry(panel: pd.DataFrame, score: str) -> pd.DataFrame:
    """与已通过训练期关卡的QM准入因子逐一比较日秩相关绝对值(冗余参照)。

    score列与准入因子均带方向已对齐(学习方向已翻转至正IC方向)，因此按
    绝对值判定冗余与core25一致。注册表内同族重复(20/60特质波动)已由人工
    决策去重，只保留60D，故此处直接与注册表全因子比较。
    """
    path = ROOT / "config/quantmind_admitted_registry.json"
    if not path.exists():
        return pd.DataFrame()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for member in cfg["factors"]:
        source = ROOT / member["signal_file"]
        column = member["score_column"]
        if not source.exists():
            print(f"admitted signal file missing: {source}", flush=True)
            continue
        old = pd.read_parquet(source, columns=["symbol", "date", column],
                              filters=[[('date', '>=', START), ('date', '<=', END)]])
        old["date"] = pd.to_datetime(old["date"])
        merged = panel[["symbol", "date", score, LABEL]].merge(
            old.rename(columns={column: "admitted"}), on=["symbol", "date"], how="inner")
        corr_col = []
        for _, day in merged.groupby("date", sort=False, observed=True):
            x = day[[score, "admitted"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x) < MIN_N or x[score].nunique() < 2 or x["admitted"].nunique() < 2:
                continue
            corr_col.append(x[score].corr(x["admitted"], method="spearman"))
        rows.append({"admitted_factor": member["factor_name"],
                     "daily_rank_corr_mean": float(np.nanmean(corr_col)) if corr_col else np.nan,
                     "daily_abs_rank_corr_mean": float(np.nanmean(np.abs(corr_col))) if corr_col else np.nan,
                     "n_days": len(corr_col)})
        print(f"admitted {len(rows):02d}/{len(cfg['factors'])} {member['factor_name']}", flush=True)
    return pd.DataFrame(rows).sort_values("daily_abs_rank_corr_mean", ascending=False)


def compute_panel(candidate: dict, start: pd.Timestamp, end: pd.Timestamp,
                  include_labels: bool = True) -> pd.DataFrame:
    """按时间窗读取行情/财务/行业输入,计算公式并做日截面 winsor_z+市值中性化。

    返回值包含 symbol/date/raw/score/LABEL 全训练窗 panel。窗越短计算越快，
    供完整评估与 --quick 前置粗筛共用，保证两条路径公式与预处理完全一致。
    """
    inputs = list(dict.fromkeys(candidate["inputs"] + ["circ_mv"]))
    formula_uses_report_operator = any(op in candidate["formula"] for op in ("FIN_LAG_REPORT", "FIN_DELTA_REPORT"))
    report_fields = [field for field in candidate["inputs"] if field in FINANCIAL_FIELDS] if formula_uses_report_operator else []
    industry_fields = [field for field in inputs if field.startswith("IND_")]
    load_inputs = [field for field in inputs if field not in report_fields and field not in industry_fields]
    load_columns = ["symbol", "date", *load_inputs]
    frame = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                            columns=list(dict.fromkeys(load_columns)),
                            filters=[[('date', '>=', start), ('date', '<=', end)]])
    frame["date"] = pd.to_datetime(frame["date"])
    if formula_uses_report_operator:
        frame = attach_report_snapshots(
            frame, report_fields, ROOT / "data/processed/financial/financial_events.parquet"
        )
    if industry_fields:
        industry = pd.read_parquet(
            ROOT / "data/processed/industry/industry_atoms_daily.parquet",
            columns=["symbol", "date", *industry_fields],
            filters=[[('date', '>=', start), ('date', '<=', end)]],
        )
        industry["date"] = pd.to_datetime(industry["date"])
        frame = frame.merge(industry, on=["symbol", "date"], how="left", validate="one_to_one")
    for col in set(inputs) & {"total_mv", "circ_mv", "pe_ttm", "pb", "ps_ttm", "turnover_rate"}:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").mask(lambda x: x == 0)
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    raw = candidate["factor_name"]
    score = f"{raw}_WINSOR_Z_SIZE_NEUTRAL"
    frame[raw] = evaluate_formula(candidate["formula"], frame)
    frame["candidate_wz"] = frame.groupby("date", observed=True)[raw].transform(winsor_z)
    frame[score] = np.nan
    for _, indices in frame.groupby("date", sort=False, observed=True).groups.items():
        frame.loc[indices, score] = neutralize_day(frame.loc[indices]).loc[indices]
    if not include_labels:
        return frame
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet", columns=["symbol", "date", LABEL],
                             filters=[[('date', '>=', start), ('date', '<=', end)]])
    labels["date"] = pd.to_datetime(labels["date"])
    return frame.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true",
                        help="近2年前置粗筛:只算 missing/rank_ic,跳过25因子与准入注册表查重,弱因子直接拒绝")
    args = parser.parse_args()
    assert_selection_period(str(START.date()), str(END.date()))
    candidate = load_candidate(args.candidate)
    out = args.output_dir or args.candidate.parent / ("quick_screen" if args.quick else "train_evaluation")
    out.mkdir(parents=True, exist_ok=True)
    # --quick 截断至近2年(2021-04-30起,含金融数据PIT所需前置量)，避免全窗7年计算开销。
    start, end = (pd.Timestamp("2021-04-30"), END) if args.quick else (START, END)
    panel = compute_panel(candidate, start, end)
    raw = candidate["factor_name"]
    score = f"{raw}_WINSOR_Z_SIZE_NEUTRAL"
    daily = daily_ic(panel, score)
    raw_values = panel[raw].replace([np.inf, -np.inf], np.nan)
    summary = {col: float(daily[col].mean()) for col in ("ic", "rank_ic")}
    summary.update({"ic_ir": float(daily.ic.mean()/daily.ic.std()) if len(daily) else float("nan"),
                    "rank_ic_ir": float(daily.rank_ic.mean()/daily.rank_ic.std()) if len(daily) else float("nan")})
    direction = "positive" if summary["rank_ic"] >= 0 else "negative"
    missing_rate = float(raw_values.isna().mean())
    panel[["symbol", "date", raw, score, LABEL]].to_parquet(out / "factor_values.parquet", index=False)
    daily.to_parquet(out / "daily_ic.parquet", index=False)
    (out / "report.json").write_text(json.dumps({
        "status": "quick_screen_complete" if args.quick else "train_evaluation_complete",
        "candidate": candidate, "score_column": score,
        "period": [str(start.date()), str(end.date())], "label_column": LABEL,
        "label_horizon_trading_days": 5, "validation_rows_read": 0, "test_rows_read": 0,
        "quality": {"rows": len(panel), "raw_missing_rate": missing_rate,
                    "infinity_count": int(np.isinf(panel[raw].to_numpy(float, na_value=np.nan)).sum()),
                    "p01": float(raw_values.quantile(.01)), "median": float(raw_values.median()),
                    "p99": float(raw_values.quantile(.99)), "max": float(raw_values.max())},
        "performance": summary, "learned_direction_train_only": direction,
        "core25_highest_redundancy": None, "admitted_registry_highest_redundancy": None,
        "train_evaluation_gates": None, "preliminary_decision": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quick:
        core = compare_core25(panel, score)
        admitted = compare_admitted_registry(panel, score)
        highest = core.iloc[0].to_dict()
        highest_admitted = admitted.iloc[0].to_dict() if not admitted.empty else None
        policy = json.loads((ROOT / "config/quantmind_candidate_policy.json").read_text(encoding="utf-8"))
        gates = policy["train_evaluation_gates"]
        decision = "eligible_for_weekly_backtest"
        if missing_rate > gates["maximum_missing_rate"]:
            decision = "rejected_excessive_missing_rate"
        elif highest["daily_abs_rank_corr_mean"] >= gates["maximum_core25_absolute_rank_correlation"]:
            decision = "rejected_duplicate_core25"
        elif highest_admitted is not None and highest_admitted["daily_abs_rank_corr_mean"] >= gates["maximum_core25_absolute_rank_correlation"]:
            decision = "rejected_duplicate_admitted"
        elif abs(summary["rank_ic"]) < gates["minimum_absolute_rank_ic"]:
            decision = "rejected_weak_train_rank_ic"
        elif abs(highest["partial_rank_ic"]) < gates["minimum_absolute_incremental_rank_ic"]:
            decision = "rejected_weak_incremental_rank_ic"
        report_path = out / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update({
            "core25_highest_redundancy": highest,
            "admitted_registry_highest_redundancy": highest_admitted,
            "train_evaluation_gates": gates, "preliminary_decision": decision,
        })
        core.to_csv(out / "core25_comparison.csv", index=False, encoding="utf-8-sig")
        if not admitted.empty:
            admitted.to_csv(out / "admitted_registry_comparison.csv", index=False, encoding="utf-8-sig")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    # quick 模式:与完整评估共享同一policy阈值,只判缺失率与弱IC两关(不查冗余);
    # 通过者才有资格跑完整评估,故quick是完整评估的真子集,不会误杀弱IC之外的原因。
    quick_policy = json.loads((ROOT / "config/quantmind_candidate_policy.json").read_text(encoding="utf-8"))
    qgates = quick_policy["train_evaluation_gates"]
    decision = "quick_passed"
    if missing_rate > qgates["maximum_missing_rate"]:
        decision = "rejected_excessive_missing_rate"
    elif abs(summary["rank_ic"]) < qgates["minimum_absolute_rank_ic"]:
        decision = "rejected_weak_train_rank_ic"
    out_report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    out_report["preliminary_decision"] = decision
    (out / "report.json").write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
