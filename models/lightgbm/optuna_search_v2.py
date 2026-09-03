"""LightGBM 多折稳健搜参：RankIC 为主，兼顾扣成本后的头部组合效用。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_LGB_DIR, TEST_PERIOD
from models.lightgbm.config import DEFAULT_LGB_PARAMS, EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND, ROLLING
from models.lightgbm.data import label_horizon, load_factor_list, prepare_data
from models.lightgbm.evaluate import calc_ic

# 研究方案冻结值：不得根据本轮验证结果事后调整。
# 如未来确需修改，应作为新实验版本并使用新的外部验证期。
STABILITY_PENALTY = 0.50
ECONOMIC_WEIGHT = 0.05
TRANSACTION_COST_BPS = 10.0
TOP_FRACTION = 0.10


def _limit_mask(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(True, index=frame.index)
    cols = (["lock_limit_up", "lock_limit_down"] if mode == "lock"
            else ["limit_up", "limit_down"])
    cols = [c for c in cols if c in frame.columns]
    return (~frame[cols].fillna(False).any(axis=1) if cols
            else pd.Series(True, index=frame.index))


def build_folds(df: pd.DataFrame, features: list[str], label: str, n_folds: int,
                fold_step: int, limit_filter: str, max_train_rows: int):
    """构造由早到晚排列的 purged walk-forward folds。"""
    dates = np.array(sorted(d for d in df["date"].unique() if d < pd.Timestamp(TEST_PERIOD[0])))
    horizon = label_horizon(label)
    folds = []
    for fold in range(n_folds):
        valid_end = len(dates) - horizon - fold * fold_step
        valid_start = valid_end - ROLLING["valid_len"]
        train_end = valid_start - horizon
        train_start = train_end - ROLLING["train_len"]
        if train_start < 0:
            raise ValueError("多折窗口历史不足")
        train_dates, valid_dates = dates[train_start:train_end], dates[valid_start:valid_end]
        tr = df[df["date"].isin(train_dates)]
        va = df[df["date"].isin(valid_dates)]
        tr = tr.loc[tr[label].notna() & np.isfinite(tr[label]) & _limit_mask(tr, limit_filter)]
        va = va.loc[va[label].notna() & np.isfinite(va[label]) & _limit_mask(va, limit_filter)]
        if len(tr) > max_train_rows:
            tr = tr.sample(max_train_rows, random_state=4200 + fold)
        period = {
            "train": [str(pd.Timestamp(train_dates[0]).date()), str(pd.Timestamp(train_dates[-1]).date())],
            "valid": [str(pd.Timestamp(valid_dates[0]).date()), str(pd.Timestamp(valid_dates[-1]).date())],
        }
        folds.append({
            "X_train": tr[features], "y_train": tr[label],
            "X_valid": va[features], "y_valid": va[label],
            "valid_dates": va["date"].to_numpy(),
            "valid_symbols": va["symbol"].astype(str).to_numpy(),
            "period": period,
        })
        print(f"[FOLD {fold}] train={len(tr):,} valid={len(va):,} period={period}")
    return list(reversed(folds))


def portfolio_utility(frame: pd.DataFrame, label: str, horizon: int,
                      top_fraction: float, cost_bps: float) -> dict[str, float]:
    """非重叠调仓点的 Top 组市场中性收益，扣除由持仓变化估算的交易成本。"""
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction 必须位于 (0, 1)")
    dates = np.array(sorted(pd.to_datetime(frame["date"].unique())))
    rebalance_dates = set(dates[::max(int(horizon), 1)])
    utilities, gross_returns, turnovers = [], [], []
    previous: set[str] = set()
    for date, group in frame.groupby("date", sort=True):
        if pd.Timestamp(date) not in rebalance_dates:
            continue
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred", label])
        if len(group) < 20:
            continue
        top = group.nlargest(max(1, int(np.ceil(len(group) * top_fraction))), "pred")
        current = set(top["symbol"].astype(str))
        turnover = 1.0 if not previous else 1.0 - len(previous & current) / max(len(current), 1)
        gross = float(top[label].mean() - group[label].mean())
        scale = max(float(group[label].std(ddof=0)), 1e-8)
        utilities.append((gross - cost_bps / 10_000.0 * turnover) / scale)
        gross_returns.append(gross)
        turnovers.append(turnover)
        previous = current
    return {
        "net_excess_utility": float(np.mean(utilities)) if utilities else -1.0,
        "gross_excess_return": float(np.mean(gross_returns)) if gross_returns else np.nan,
        "turnover": float(np.mean(turnovers)) if turnovers else 1.0,
        "rebalance_count": int(len(utilities)),
    }


def _sample_params(trial) -> dict:
    """百万行截面数据的保守、正则化搜索空间。"""
    depth = trial.suggest_int("max_depth", 4, 10)
    return {
        "objective": "regression", "metric": "mse", "verbosity": -1,
        "seed": 42, "feature_fraction_seed": 42, "bagging_seed": 42,
        "data_random_seed": 42, "deterministic": True, "force_col_wise": True,
        "num_threads": 8,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth": depth,
        "num_leaves": trial.suggest_int("num_leaves", 15, min(255, 2 ** depth - 1)),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 5000, log=True),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-6, 1.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 1e3, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 1e3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.60, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.60, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
    }


def objective_factory(folds, label: str):
    horizon = label_horizon(label)

    def objective(trial):
        params = _sample_params(trial)
        rank_ics, economics, turnovers, iterations = [], [], [], []
        for fold_id, fold in enumerate(folds):
            train_set = lgb.Dataset(fold["X_train"], label=fold["y_train"])
            valid_set = lgb.Dataset(fold["X_valid"], label=fold["y_valid"], reference=train_set)
            model = lgb.train(
                params, train_set, num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[valid_set], valid_names=["valid"],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            pred = model.predict(fold["X_valid"], num_iteration=model.best_iteration)
            frame = pd.DataFrame({"date": fold["valid_dates"], "symbol": fold["valid_symbols"],
                                  "pred": pred, label: fold["y_valid"].to_numpy()})
            daily = calc_ic(frame, "pred", label)
            rank_ics.append(float(daily["rank_ic"].mean()) if not daily.empty else -1.0)
            econ = portfolio_utility(
                frame, label, horizon, TOP_FRACTION, TRANSACTION_COST_BPS
            )
            economics.append(econ["net_excess_utility"])
            turnovers.append(econ["turnover"])
            iterations.append(int(model.best_iteration))
            trial.report(
                float(np.mean(rank_ics) + ECONOMIC_WEIGHT * np.mean(economics)), fold_id
            )
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        score = float(np.mean(rank_ics) - STABILITY_PENALTY * np.std(rank_ics)
                      + ECONOMIC_WEIGHT * np.mean(economics))
        trial.set_user_attr("fold_rank_ic", rank_ics)
        trial.set_user_attr("fold_net_excess_utility", economics)
        trial.set_user_attr("fold_turnover", turnovers)
        trial.set_user_attr("mean_rank_ic", float(np.mean(rank_ics)))
        trial.set_user_attr("std_rank_ic", float(np.std(rank_ics)))
        trial.set_user_attr("worst_rank_ic", float(np.min(rank_ics)))
        trial.set_user_attr("best_iterations", iterations)
        return score
    return objective


def _production_params(sampled: dict) -> dict:
    """消除 LightGBM 别名冲突，形成 train.py 可直接读取的参数。"""
    params = {**DEFAULT_LGB_PARAMS, **sampled}
    params["colsample_bytree"] = params.pop("feature_fraction")
    params["subsample"] = params.pop("bagging_fraction")
    params["subsample_freq"] = params.pop("bagging_freq")
    return params


def _promote(record: dict) -> Path:
    path = MODELS_LGB_DIR / "best_params.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "label" in data and "params" in data:
        data = {data["label"]: data}
    data[record["label"]] = {
        "label": record["label"], "n_trials": record["n_trials"],
        "objective": record["objective"], "best_trial": record["best_trial"],
        "best_rank_ic": record["mean_rank_ic"],
        "best_robust_score": record["best_robust_score"], "params": record["params"],
        "created_by": "optuna_search_v2.py --promote",
        "candidate_file": record["candidate_file"],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="多折稳健 LightGBM Optuna")
    parser.add_argument("--label", default="label_ret_20")
    parser.add_argument("--factor-list", default="passed_factor_cols_v2.json")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-step", type=int, default=120)
    parser.add_argument("--max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--limit-filter", choices=("all", "lock", "none"), default="all")
    parser.add_argument("--promote", action="store_true",
                        help="显式写入生产 best_params.json；默认只产生候选")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick and args.promote:
        parser.error("quick 冒烟结果禁止晋级生产")

    factors = load_factor_list(args.factor_list, label_col=args.label)
    if not factors:
        raise ValueError(f"V2因子清单为空: {args.factor_list}")
    df, features = prepare_data(args.label, factors)
    folds = build_folds(df, features, args.label, 2 if args.quick else args.n_folds,
                        args.fold_step, args.limit_filter,
                        min(args.max_train_rows, 200_000) if args.quick else args.max_train_rows)
    import optuna
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.optimize(objective_factory(folds, args.label),
                   n_trials=3 if args.quick else args.n_trials)
    best = study.best_trial
    MODELS_LGB_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.quick else ""
    output = MODELS_LGB_DIR / f"best_params_v2{suffix}.json"
    record = {
        "version": "v2_multifold_economic", "label": args.label,
        "factor_list": args.factor_list, "n_features": len(features),
        "n_trials": len(study.trials),
        "objective": "mean_rank_ic - stability_penalty*std_rank_ic + economic_weight*net_top_utility",
        "cost_bps": TRANSACTION_COST_BPS, "top_fraction": TOP_FRACTION,
        "economic_weight": ECONOMIC_WEIGHT, "stability_penalty": STABILITY_PENALTY,
        "scoring_policy_frozen": True,
        "best_trial": best.number, "best_robust_score": best.value,
        "fold_rank_ic": best.user_attrs["fold_rank_ic"],
        "fold_net_excess_utility": best.user_attrs["fold_net_excess_utility"],
        "fold_turnover": best.user_attrs["fold_turnover"],
        "mean_rank_ic": best.user_attrs["mean_rank_ic"],
        "std_rank_ic": best.user_attrs["std_rank_ic"],
        "worst_rank_ic": best.user_attrs["worst_rank_ic"],
        "best_iterations": best.user_attrs["best_iterations"],
        "folds": [f["period"] for f in folds], "params": _production_params(best.params),
        "candidate_file": output.name, "promoted": bool(args.promote),
        "test_data_used": False,
        "warning": "候选参数须经完整滚动验证；冻结测试集不参与搜索。",
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
        MODELS_LGB_DIR / f"optuna_trials_v2{suffix}.csv", index=False)
    if args.promote:
        print(f"[OK] 已显式晋级生产参数: {_promote(record)}")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"[OK] V2候选参数: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
