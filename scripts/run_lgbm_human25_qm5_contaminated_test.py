"""One guarded secondary test run for the frozen PIT/purged LightGBM experiment.

The global test interval was evaluated by an older label_ret_20 model, so this
run is permanently labelled contaminated_secondary_test and never presented as
an unbiased final test.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.config import PortfolioConfig
from backtest.engine import run_portfolio_backtest
from backtest.run_backtest import MARKET_COLUMNS, _attach_open_limit_state
from config.settings import TEST_PERIOD
from scripts.evaluate_quantmind_candidate import compute_panel
from scripts.run_lgbm_human25_qm5_weekly import (
    FEATURE_FILE, FEATURE_LIST, QM5, _score_name, preprocess_human_part,
)

START, END = map(pd.Timestamp, TEST_PERIOD)
BUFFER_START = pd.Timestamp("2024-07-01")
VALID_REPORT = ROOT / "reports/lightgbm/human25_qm5_weekly_v2_pit_purged/report.json"
OUT = ROOT / "reports/lightgbm/human25_qm5_weekly_v2_pit_purged/contaminated_secondary_test"
TEST_FEATURES = OUT / "test_features.parquet"
MODEL_LOCK = OUT / "test_evaluation_audit.json"
OLD_GLOBAL_LOCK = ROOT / "reports/portfolio/final_test_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze() -> dict:
    validation = json.loads(VALID_REPORT.read_text(encoding="utf-8"))
    old = json.loads(OLD_GLOBAL_LOCK.read_text(encoding="utf-8")) if OLD_GLOBAL_LOCK.exists() else None
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "contaminated_secondary_test",
        "reason": "same test interval was previously evaluated by another model/label",
        "prior_global_test_audit": old,
        "features": json.loads(FEATURE_LIST.read_text(encoding="utf-8")),
        "params": validation["tuning"]["params"],
        "num_boost_round": validation["tuning"]["best_iteration"],
        "portfolio": {"topk": 50, "buffer": 30, "max_drop": 10,
                      "rebalance_rule": "calendar_week_end",
                      "execution": "next_trading_day_open",
                      "buy_cost_bps": 10, "sell_cost_bps": 10,
                      "hard_turnover_cap": "disabled"},
        "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [
            FEATURE_LIST, VALID_REPORT, ROOT / "backtest/engine.py",
            ROOT / "backtest/alpha_signal.py", ROOT / "backtest/config.py",
            ROOT / "scripts/run_lgbm_human25_qm5_weekly.py",
        ]},
        "test_metrics_seen_before_freeze": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_test_features(force: bool = False) -> list[str]:
    feature_cfg = json.loads(FEATURE_LIST.read_text(encoding="utf-8"))
    features = feature_cfg["factors"]
    if TEST_FEATURES.exists() and not force:
        return features
    core = json.loads((ROOT / "config/human_core_25_v1.json").read_text(encoding="utf-8"))
    human = [x["factor_name"] for x in core["factors"]]
    existing = [x for x in human if x != "MAX_DD60_V2"]
    sidecars = []
    for candidate in QM5:
        side = OUT / f"{candidate['factor_name']}_feature_only.parquet"
        print(f"[TEST FEATURE-ONLY] {candidate['factor_name']}", flush=True)
        panel = compute_panel(candidate, BUFFER_START, END, include_labels=False)
        col = _score_name(candidate)
        panel = panel[panel["date"] >= START]
        panel[["symbol", "date", col]].to_parquet(side, index=False)
        sidecars.append(side)
        del panel
        gc.collect()

    pieces = []
    for year in range(START.year, END.year + 1):
        lo, hi = max(START, pd.Timestamp(f"{year}-01-01")), min(END, pd.Timestamp(f"{year}-12-31"))
        filt = [[("date", ">=", lo), ("date", "<=", hi)]]
        part = pd.read_parquet(ROOT / "data/processed/factors.parquet",
                               columns=["symbol", "date", *existing], filters=filt)
        if part.empty:
            continue
        part["date"] = pd.to_datetime(part["date"])
        fix = pd.read_parquet(ROOT / "data/processed/reversal_momentum_fixes_v2.parquet",
                              columns=["symbol", "date", "MAX_DD60_V2"], filters=filt)
        fix["date"] = pd.to_datetime(fix["date"])
        basic = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                                columns=["symbol", "date", "circ_mv"], filters=filt)
        basic["date"] = pd.to_datetime(basic["date"])
        basic["circ_mv"] = pd.to_numeric(basic["circ_mv"], errors="coerce").mask(lambda x: x <= 0)
        basic["log_mv"] = np.log(basic["circ_mv"])
        industry = pd.read_parquet(ROOT / "data/processed/industry/industry_atoms_daily.parquet",
                                   columns=["symbol", "date", "industry_code"], filters=filt)
        industry["date"] = pd.to_datetime(industry["date"])
        industry = industry.rename(columns={"industry_code": "industry"})
        part = part.merge(fix, on=["symbol", "date"], how="left", validate="one_to_one")
        part = part.merge(basic[["symbol", "date", "log_mv"]],
                          on=["symbol", "date"], how="left", validate="one_to_one")
        part = part.merge(industry, on=["symbol", "date"], how="left", validate="one_to_one")
        part = preprocess_human_part(part, human)
        for side, candidate in zip(sidecars, QM5):
            col = _score_name(candidate)
            q = pd.read_parquet(side, filters=filt); q["date"] = pd.to_datetime(q["date"])
            part = part.merge(q, on=["symbol", "date"], how="left", validate="one_to_one")
        part = part[["symbol", "date", "industry", "log_mv", *features]]
        pieces.append(part)
        print(f"[TEST FEATURE] {year}: {len(part):,}", flush=True)
    test = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    test.to_parquet(TEST_FEATURES, index=False)
    return features


def train_once_and_score(features: list[str], manifest: dict) -> pd.DataFrame:
    pretest = pd.read_parquet(FEATURE_FILE)
    pretest["date"] = pd.to_datetime(pretest["date"])
    calendar = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                               columns=["date"], filters=[[('date', '>=', pd.Timestamp('2024-11-01')),
                                                          ('date', '<=', START)]])
    dates = pd.DatetimeIndex(sorted(calendar["date"].dropna().unique()))
    first = int(dates.searchsorted(START, side="left"))
    cutoff = pd.Timestamp(dates[first - 6])
    fit = pretest[(pretest["date"] <= cutoff) & pretest["label_ret_5"].notna()]
    if fit.empty or fit["date"].max() >= START:
        raise RuntimeError("测试前训练边界异常")
    model = lgb.train(manifest["params"], lgb.Dataset(fit[features], label=fit["label_ret_5"]),
                      num_boost_round=int(manifest["num_boost_round"]))
    test = pd.read_parquet(TEST_FEATURES)
    pred = test[["symbol", "date"]].copy()
    pred["pred"] = model.predict(test[features])
    model.save_model(str(OUT / "final_refit_model.txt"))
    pred.to_parquet(OUT / "test_scores.parquet", index=False)
    (OUT / "training_audit.json").write_text(json.dumps({
        "fit_rows": len(fit), "fit_start": str(fit["date"].min().date()),
        "fit_end_purged": str(cutoff.date()), "test_start": str(START.date()),
        "label_horizon": 5, "purge_trading_days": 5,
        "test_labels_loaded_for_training": False,
        "test_feature_has_label_columns": any(c.startswith("label_") for c in test.columns),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return pred


def backtest(pred: pd.DataFrame) -> dict:
    market = pd.read_parquet(ROOT / "data/processed/basic_cleaned_with_extra_by_date.parquet",
                             columns=MARKET_COLUMNS,
                             filters=[[('date', '>=', START - pd.Timedelta(days=240)),
                                       ('date', '<=', END + pd.Timedelta(days=10))]])
    market["date"] = pd.to_datetime(market["date"]); market = _attach_open_limit_state(market)
    pit = pd.read_parquet(ROOT / "data/processed/industry/industry_atoms_daily.parquet",
                          columns=["symbol", "date", "industry_code"],
                          filters=[[('date', '>=', START), ('date', '<=', END)]])
    pit["date"] = pd.to_datetime(pit["date"]); pit = pit.rename(columns={"industry_code": "industry"})
    predictions = pred.merge(pit, on=["symbol", "date"], how="inner", validate="one_to_one")
    predictions = predictions.merge(market[["symbol", "date", "circ_mv"]],
                                    on=["symbol", "date"], how="inner", validate="one_to_one")
    benchmark = pd.read_parquet(ROOT / "data/raw/daily/daily_sh.parquet",
                                columns=["symbol", "date", "close"],
                                filters=[[('symbol', '==', 'SH000300'),
                                          ('date', '>=', START - pd.Timedelta(days=10)),
                                          ('date', '<=', END + pd.Timedelta(days=10))]])
    config = PortfolioConfig(topk=50, buffer=30, max_drop=10,
                             rebalance_rule="calendar_week_end", rebalance_days=5,
                             buy_cost=.001, sell_cost=.001,
                             max_rebalance_turnover=float("inf"),
                             use_mean_variance_optimizer=False, use_dynamic_exposure=False)
    result = run_portfolio_backtest(predictions, market, benchmark, config)
    if result.trades.empty or result.nav["is_rebalance"].sum() == 0:
        raise RuntimeError("测试回测无成交，拒绝完成")
    result.nav.to_parquet(OUT / "nav.parquet", index=False)
    result.trades.to_parquet(OUT / "trades.parquet", index=False)
    result.holdings.to_parquet(OUT / "holdings.parquet", index=False)
    return result.metrics


def main() -> int:
    if MODEL_LOCK.exists():
        raise RuntimeError(f"本模型测试已经执行并锁定: {MODEL_LOCK}")
    manifest = freeze()
    features = build_test_features()
    pred = train_once_and_score(features, manifest)
    metrics = backtest(pred)
    report = {"status": "complete", "classification": "contaminated_secondary_test",
              "unbiased_final_test": False, "period": list(TEST_PERIOD),
              "metrics": metrics, "test_labels_used_for_training_or_tuning": False,
              "prior_global_test_lock_preserved": OLD_GLOBAL_LOCK.exists()}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MODEL_LOCK.write_text(json.dumps({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "contaminated_secondary_test", "model": "human25_qm5_v2_pit_purged",
        "label": "label_ret_5", "period": list(TEST_PERIOD),
        "report": str((OUT / "report.json").relative_to(ROOT)).replace("\\", "/"),
        "rerun_allowed": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
