"""Build the frozen human25+QM5 feature set, tune LightGBM, and weekly-backtest it.

The final test period is never loaded.  The official validation interval is split
once into tuning and holdout portions so the reported weekly backtest is not the
same slice used by Optuna.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.run_backtest import _attach_open_limit_state
from backtest.weekly_factor_engine import run_weekly_factor_backtest
from config.settings import TRAIN_PERIOD, VALID_PERIOD
from models.lightgbm.evaluate import calc_ic
from models.lightgbm.optuna_search_v2 import (
    ECONOMIC_WEIGHT, STABILITY_PENALTY, TOP_FRACTION, TRANSACTION_COST_BPS,
    _production_params, _sample_params, portfolio_utility,
)
from scripts.evaluate_quantmind_candidate import compute_panel

OUT = ROOT / "reports" / "lightgbm" / "human25_qm5_weekly_v2_pit_purged"
FEATURE_FILE = ROOT / "data" / "processed" / "features_human25_qm5_v2_pit.parquet"
FEATURE_LIST = ROOT / "data" / "processed" / "human25_qm5_ret5_v2_pit.json"
TUNE_END = pd.Timestamp("2024-06-28")
HOLDOUT_START = pd.Timestamp("2024-07-01")
END = pd.Timestamp(VALID_PERIOD[1])

QM5 = [
    {
        "factor_name": "QM_DS_LIQUIDITY_AMPLIFICATION_TURNOVER_RETVOL",
        "formula": "SAFE_DIV(TS_MEAN($turnover_rate,5),TS_STD(TS_PCT_CHANGE($close,1),20)*SAFE_DIV(TS_MEAN($turnover_rate,20),TS_MEAN($turnover_rate,5)))",
        "inputs": ["turnover_rate", "close"], "lookback": 21,
        "availability": "after_close", "direction": "negative",
        "economic_rationale": "低流动性放大风险溢价。",
    },
    {
        "factor_name": "QM_DS_IDIOVOL_60D", "formula": "TS_STD($IND_RESID_RET_1D,60)",
        "inputs": ["IND_RESID_RET_1D"], "lookback": 60,
        "availability": "after_close", "direction": "negative",
        "economic_rationale": "长期特质波动风险。",
    },
    {
        "factor_name": "QM_COMPOSITE_QUALITY_LOW_CROWDING_LOWVOL30",
        "formula": "SIGN($roe)*LOG1P(ABS($roe))-LOG1P($turnover_rate)-30*TS_STD(TS_PCT_CHANGE($close,1),20)",
        "inputs": ["roe", "turnover_rate", "close"], "lookback": 21,
        "availability": "after_close", "direction": "positive",
        "economic_rationale": "质量、低拥挤和低波动复合。",
    },
    {
        "factor_name": "QM_DS_IND_RESID_VOL_20", "formula": "TS_STD($IND_RESID_RET_1D,20)",
        "inputs": ["IND_RESID_RET_1D"], "lookback": 20,
        "availability": "after_close", "direction": "negative",
        "economic_rationale": "短期特质波动风险；与60日版本高度相关，仅用于本次指定实验。",
    },
    {
        "factor_name": "QM_COMPOSITE_QUALITY_LOW_CROWDING_LOWVOL",
        "formula": "SIGN($roe)*LOG1P(ABS($roe))-LOG1P($turnover_rate)-20*TS_STD(TS_PCT_CHANGE($close,1),20)",
        "inputs": ["roe", "turnover_rate", "close"], "lookback": 21,
        "availability": "after_close", "direction": "positive",
        "economic_rationale": "低波动惩罚20倍的近门槛版本；仅用于本次指定实验。",
    },
]


def _score_name(candidate: dict) -> str:
    return f"{candidate['factor_name']}_WINSOR_Z_SIZE_NEUTRAL"


def preprocess_human_part(part: pd.DataFrame, human: list[str]) -> pd.DataFrame:
    """Daily winsorize + PIT-industry/size neutralize + z-score."""
    processed_days = []
    for _, day in part.groupby("date", sort=False, observed=True):
        day = day.copy()
        numeric = day[human].replace([np.inf, -np.inf], np.nan).astype(float)
        q = numeric.quantile([.01, .99])
        numeric = numeric.clip(lower=q.iloc[0], upper=q.iloc[1], axis=1)
        valid_meta = day["industry"].notna() & day["log_mv"].notna()
        if valid_meta.sum() >= 30:
            base_columns = list(pd.get_dummies(
                day.loc[valid_meta, "industry"].astype(str), dtype=float
            ).columns)
            for col in human:
                ok = valid_meta & numeric[col].notna()
                if ok.sum() < 30:
                    continue
                xi = pd.get_dummies(day.loc[ok, "industry"].astype(str), dtype=float)
                xi = xi.reindex(columns=base_columns, fill_value=0.0)
                xi["log_mv"] = day.loc[ok, "log_mv"].astype(float)
                yy, xx = numeric.loc[ok, col].to_numpy(), xi.to_numpy()
                numeric.loc[ok, col] = yy - xx @ np.linalg.lstsq(xx, yy, rcond=None)[0]
        mean, std = numeric.mean(), numeric.std(ddof=1)
        day[human] = ((numeric - mean) / std.replace(0, np.nan)).astype("float32")
        processed_days.append(day)
    return pd.concat(processed_days, ignore_index=True)


def build_feature_file(force: bool = False) -> list[str]:
    core = json.loads((ROOT / "config/human_core_25_v1.json").read_text(encoding="utf-8"))
    human = [x["factor_name"] for x in core["factors"]]
    qm_cols = [_score_name(x) for x in QM5]
    all_features = human + qm_cols
    FEATURE_LIST.write_text(json.dumps({
        "dataset_id": "human25_qm5_v2_pit_purged", "factors": all_features,
        "human_factor_count": 25, "qm_factor_count": 5,
        "criteria": {"label": "label_ret_5", "test_data_used": False},
        "qm_warning": "QM5 includes one duplicate-window and one near-threshold experimental factor by user instruction.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if FEATURE_FILE.exists() and not force:
        return all_features

    OUT.mkdir(parents=True, exist_ok=True)
    sidecars = []
    for candidate in QM5:
        side = OUT / f"{candidate['factor_name']}_through_valid.parquet"
        if not side.exists() or force:
            print(f"[FEATURE] compute {candidate['factor_name']} through {END.date()}", flush=True)
            panel = compute_panel(candidate, pd.Timestamp(TRAIN_PERIOD[0]), END)
            col = _score_name(candidate)
            panel[["symbol", "date", col]].to_parquet(side, index=False)
            del panel
            gc.collect()
        sidecars.append(side)

    base_path = ROOT / "data/processed/factors.parquet"
    basic_path = ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet"
    industry_path = ROOT / "data/processed/industry/industry_atoms_daily.parquet"
    fix_path = ROOT / "data/processed/reversal_momentum_fixes_v2.parquet"
    labels_path = ROOT / "data/processed/labels.parquet"
    existing_human = [x for x in human if x != "MAX_DD60_V2"]
    meta = ["symbol", "date", "industry", "log_mv"]
    writer = None
    try:
        for year in range(pd.Timestamp(TRAIN_PERIOD[0]).year, END.year + 1):
            lo, hi = pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), END)
            filt = [[("date", ">=", lo), ("date", "<=", hi)]]
            part = pd.read_parquet(base_path, columns=["symbol", "date", *existing_human], filters=filt)
            if part.empty:
                print(f"[FEATURE] skip empty {year}", flush=True)
                continue
            part["date"] = pd.to_datetime(part["date"])
            basic = pd.read_parquet(basic_path, columns=["symbol", "date", "circ_mv"], filters=filt)
            basic["date"] = pd.to_datetime(basic["date"])
            basic["circ_mv"] = pd.to_numeric(basic["circ_mv"], errors="coerce").mask(lambda x: x <= 0)
            basic["log_mv"] = np.log(basic["circ_mv"])
            industry = pd.read_parquet(industry_path, columns=["symbol", "date", "industry_code"], filters=filt)
            industry["date"] = pd.to_datetime(industry["date"])
            industry = industry.rename(columns={"industry_code": "industry"})
            part = part.merge(basic[["symbol", "date", "log_mv"]],
                              on=["symbol", "date"], how="left", validate="one_to_one")
            part = part.merge(industry, on=["symbol", "date"], how="left", validate="one_to_one")
            fix = pd.read_parquet(fix_path, columns=["symbol", "date", "MAX_DD60_V2"], filters=filt)
            fix["date"] = pd.to_datetime(fix["date"])
            part = part.merge(fix, on=["symbol", "date"], how="left", validate="one_to_one")
            # Rebuild all 25 human factors from raw values using historical PIT
            # industry membership.  The old features.parquet used a latest-industry
            # snapshot and is intentionally not reused here.
            part = preprocess_human_part(part, human)
            for side, col in zip(sidecars, qm_cols):
                q = pd.read_parquet(side, filters=filt)
                q["date"] = pd.to_datetime(q["date"])
                part = part.merge(q, on=["symbol", "date"], how="left", validate="one_to_one")
            lab = pd.read_parquet(labels_path, columns=["symbol", "date", "label_ret_5"], filters=filt)
            lab["date"] = pd.to_datetime(lab["date"])
            part = part.merge(lab, on=["symbol", "date"], how="left", validate="one_to_one")
            for col in all_features:
                part[col] = pd.to_numeric(part[col], errors="coerce").astype("float32")
            part = part[meta + all_features + ["label_ret_5"]].sort_values(["date", "symbol"])
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(FEATURE_FILE, table.schema, compression="zstd")
            writer.write_table(table)
            print(f"[FEATURE] wrote {year}: {len(part):,}", flush=True)
            del part, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    return all_features


def _daily_rank_ic(date, pred, label) -> float:
    frame = pd.DataFrame({"date": date, "pred": pred, "label_ret_5": label})
    daily = calc_ic(frame, "pred", "label_ret_5")
    return float(daily["rank_ic"].mean()) if not daily.empty else -1.0


def tune_and_fit(features: list[str], n_trials: int) -> tuple[pd.DataFrame, dict]:
    import optuna
    data = pd.read_parquet(FEATURE_FILE)
    data["date"] = pd.to_datetime(data["date"])
    dates = pd.DatetimeIndex(sorted(data["date"].dropna().unique()))
    def cutoff_before(next_start: pd.Timestamp, horizon: int = 5) -> pd.Timestamp:
        first = int(dates.searchsorted(next_start, side="left"))
        if first <= horizon:
            raise ValueError("历史不足以执行标签边界隔离")
        return pd.Timestamp(dates[first - horizon - 1])
    train_cutoff = min(pd.Timestamp(TRAIN_PERIOD[1]), cutoff_before(pd.Timestamp(VALID_PERIOD[0])))
    tune_cutoff = min(TUNE_END, cutoff_before(HOLDOUT_START))
    train = data[(data.date >= pd.Timestamp(TRAIN_PERIOD[0])) & (data.date <= train_cutoff)]
    tune = data[(data.date >= pd.Timestamp(VALID_PERIOD[0])) & (data.date <= tune_cutoff)]
    holdout = data[(data.date >= HOLDOUT_START) & (data.date <= END)]
    train = train[train.label_ret_5.notna()].copy(); tune = tune[tune.label_ret_5.notna()].copy()
    if len(train) > 1_500_000:
        train_fit = train.sample(1_500_000, random_state=42)
    else:
        train_fit = train
    Xtr, ytr = train_fit[features], train_fit.label_ret_5
    Xva, yva = tune[features], tune.label_ret_5

    def objective(trial):
        params = _sample_params(trial)
        model = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=700,
                          valid_sets=[lgb.Dataset(Xva, label=yva)],
                          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        pred = model.predict(Xva, num_iteration=model.best_iteration)
        ric = _daily_rank_ic(tune.date.to_numpy(), pred, yva.to_numpy())
        econ_frame = tune[["date", "symbol", "label_ret_5"]].copy(); econ_frame["pred"] = pred
        econ = portfolio_utility(econ_frame, "label_ret_5", 5, TOP_FRACTION, TRANSACTION_COST_BPS)
        score = ric + ECONOMIC_WEIGHT * econ["net_excess_utility"]
        trial.set_user_attr("rank_ic", ric); trial.set_user_attr("turnover", econ["turnover"])
        trial.set_user_attr("best_iteration", int(model.best_iteration)); return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
    study.optimize(objective, n_trials=n_trials)
    params = _production_params(study.best_trial.params)
    best_iter = int(study.best_trial.user_attrs["best_iteration"])
    # Refit on training + tuning validation; holdout remains untouched until prediction.
    fit = pd.concat([train, tune], ignore_index=True)
    model = lgb.train(params, lgb.Dataset(fit[features], label=fit.label_ret_5), num_boost_round=max(best_iter, 20))
    pred = model.predict(holdout[features])
    scores = holdout[["symbol", "date"]].copy(); scores["lgb_score"] = pred
    holdout_daily = calc_ic(
        pd.DataFrame({"date": holdout["date"].to_numpy(), "lgb_score": pred,
                      "label_ret_5": holdout["label_ret_5"].to_numpy()}),
        "lgb_score", "label_ret_5",
    )
    model.save_model(str(OUT / "lightgbm_human25_qm5.txt"))
    importance = pd.DataFrame({"factor": features, "gain": model.feature_importance("gain"),
                               "split": model.feature_importance("split")}).sort_values("gain", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    record = {"best_trial": study.best_trial.number, "best_score": study.best_value,
              "tune_rank_ic": study.best_trial.user_attrs["rank_ic"],
              "tune_turnover_proxy": study.best_trial.user_attrs["turnover"],
              "best_iteration": best_iter, "params": params,
              "train_period": [TRAIN_PERIOD[0], str(train_cutoff.date())],
              "tune_period": [VALID_PERIOD[0], str(tune_cutoff.date())],
              "label_boundary_purge_trading_days": 5,
              "holdout_period": [str(HOLDOUT_START.date()), VALID_PERIOD[1]],
              "holdout_ic": float(holdout_daily["ic"].mean()),
              "holdout_rank_ic": float(holdout_daily["rank_ic"].mean()),
              "holdout_rank_ic_ir": float(holdout_daily["rank_ic"].mean() / holdout_daily["rank_ic"].std()),
              "test_data_used": False}
    return scores, record


def weekly_backtest(scores: pd.DataFrame) -> dict:
    start, end = HOLDOUT_START, END
    market_cols = ["symbol", "date", "open", "close", "factor", "volume"]
    filt = [[("date", ">=", start), ("date", "<=", end)]]
    market = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                             columns=market_cols, filters=filt)
    market = _attach_open_limit_state(market)
    benchmark = pd.read_parquet(ROOT / "data/raw/daily/daily_sh.parquet",
                                columns=["symbol", "date", "close"],
                                filters=[[('symbol', '==', 'SH000300'), ('date', '>=', start), ('date', '<=', end)]])
    result = run_weekly_factor_backtest(scores, market, benchmark, score_col="lgb_score",
                                        start=start, end=end, cost_bps=10, long_quantile=.10)
    case = OUT / "weekly_10bps_holdout"; case.mkdir(parents=True, exist_ok=True)
    result.nav.to_parquet(case / "nav.parquet", index=False)
    result.trades.to_parquet(case / "trades.parquet", index=False)
    result.holdings.to_parquet(case / "holdings.parquet", index=False)
    result.schedule.to_parquet(case / "schedule.parquet", index=False)
    return result.metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    features = build_feature_file(args.force_features)
    scores, tuning = tune_and_fit(features, args.n_trials)
    scores.to_parquet(OUT / "validation_holdout_scores.parquet", index=False)
    metrics = weekly_backtest(scores)
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "feature_count": len(features), "human_count": 25, "qm_count": 5,
              "label": "label_ret_5", "tuning": tuning, "weekly_10bps_holdout": metrics,
              "test_data_used": False,
              "warning": "This is a validation experiment, not the final locked test result."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
