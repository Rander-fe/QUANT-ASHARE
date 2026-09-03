# -*- coding: utf-8 -*-
"""LightGBM 超参搜索（Optuna）。

搜索空间借鉴 Qlib 官方 hyperparameter 脚本
（microsoft/qlib，examples/hyperparameter/LightGBM/optuna_lightgbm.py）：

流程：
    1. 用精选因子清单（passed_factor_cols.json，25 因子）加载预处理产物
    2. 抽取验证集前一段作为"快速搜索窗口"：train = 验证集前 750 交易日，
       valid = 验证集前 60 交易日（即在验证集区间内部再切，训练集绝不触碰测试集）
    3. 每个 trial 用 TPE 采样超参，LightGBM 早停训练，以 valid IC 为目标
    4. 搜索完成后回放最优 trial 的完整滚动重训（train.py 全量滚动），
       用验证集 IC/ICIR 评估
    5. 最优超参落盘 models/lightgbm/best_params.json，供 train.py 使用

用法：
    python models/lightgbm/optuna_search.py                  # 默认 30 trials
    python models/lightgbm/optuna_search.py --n-trials 60
    python models/lightgbm/optuna_search.py --quick          # 3 trials 冒烟
    python models/lightgbm/optuna_search.py --label label_ret_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    DATA_PROCESSED,
    MODELS_LGB_DIR,
    TEST_PERIOD,
)
from models.lightgbm.config import (
    DEFAULT_LGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    LABEL_COL,
    NUM_BOOST_ROUND,
    ROLLING,
)
from models.lightgbm.data import label_horizon, load_factor_list, prepare_data
from models.lightgbm.evaluate import calc_ic, summarize_ic


def _quick_split(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                 train_days: int, valid_days: int, limit_filter: str = "all"):
    """切出快速搜索用的 train/valid 窗口（两者都在测试集之前）。

    铁律说明：
        - 测试集（2025-01 之后）完全不参与搜索；
        - valid 段取验证集末尾（如 2024 下半年），train 段取 valid 之前
          最近 train_days 个交易日（会自然延伸到训练集，无未来函数）。
    窗口较滚动重训小（750/60 天），保证每个 trial 能在数分钟内完成。
    """
    dates = np.sort(df["date"].unique())
    # 只用测试集之前的日期
    before_test = [d for d in dates if d < pd.Timestamp(TEST_PERIOD[0])]
    if len(before_test) < train_days + valid_days + 1:
        raise ValueError(
            f"测试集之前交易日不足: {len(before_test)} < {train_days + valid_days + 1}"
        )

    purge_days = label_horizon(label_col)
    # 测试集前也留出隔离带，保证搜索阶段完全不使用会延伸进测试集的标签。
    va_end = len(before_test) - purge_days
    va_start = va_end - valid_days
    tr_end = va_start - purge_days
    tr_start = tr_end - train_days
    if tr_start < 0:
        raise ValueError("加入标签隔离期后历史交易日不足")
    va_dates = pd.to_datetime(before_test[va_start:va_end])
    tr_dates = pd.to_datetime(before_test[tr_start:tr_end])

    train_df = df[df["date"].isin(tr_dates)]
    valid_df = df[df["date"].isin(va_dates)]

    # 封板日过滤（与 train.py 的 _limit_mask 保持一致，仅剔除涨停+跌停）
    def _limit_mask(d: pd.DataFrame) -> pd.Series:
        cols = ["limit_up", "limit_down"]
        if not set(cols) <= set(d.columns):
            return pd.Series(True, index=d.index)
        return ~d[cols].fillna(False).any(axis=1)

    y_tr = train_df[label_col]
    y_va = valid_df[label_col]
    keep_tr = y_tr.notna() & np.isfinite(y_tr) & _limit_mask(train_df)
    keep_va = y_va.notna() & np.isfinite(y_va) & _limit_mask(valid_df)

    X_tr = train_df[feature_cols][keep_tr]
    y_tr = y_tr[keep_tr]
    X_va = valid_df[feature_cols][keep_va]
    y_va = y_va[keep_va]

    # 把 index 设为日期，供 objective 内按日计算 IC
    X_tr.index = train_df["date"][keep_tr].values
    X_va.index = valid_df["date"][keep_va].values

    print(f"[INFO] 搜索窗口: train {tr_dates[0].date()}~{tr_dates[-1].date()} "
          f"({len(X_tr):,} 样本) | valid {va_dates[0].date()}~{va_dates[-1].date()} "
          f"({len(X_va):,} 样本) | 边界隔离={purge_days} 交易日")
    return X_tr, y_tr, X_va, y_va


def objective_factory(X_tr, y_tr, X_va, y_va, label_col: str):
    """返回 Optuna objective。目标 = 验证窗口 RankIC（更稳健）"""

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "seed": 42,
            "num_threads": 8,
            # Qlib hyperparameter 脚本的搜索区间（示例）
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": 1,
            "min_data_in_bin": 1,
        }

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=0)],
        )

        # 用早停迭代数在 valid 上预测，评估 RankIC
        y_pred = model.predict(X_va, num_iteration=model.best_iteration or NUM_BOOST_ROUND)
        valid_df = pd.DataFrame(
            {"date": X_va.index, "pred": y_pred, label_col: y_va.values}
        )
        daily = calc_ic(valid_df, "pred", label_col)
        summary = summarize_ic(daily)
        return summary["rank_ic_mean"]

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGBM 超参搜索（Optuna + 验证集）")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials 数")
    parser.add_argument("--quick", action="store_true", help="冒烟：3 trials")
    parser.add_argument("--label", default=LABEL_COL, help="标签列")
    parser.add_argument("--limit-filter", choices=("all", "lock", "none"), default="all",
                        help="封板日过滤（默认 all）")
    args = parser.parse_args()

    n_trials = 3 if args.quick else args.n_trials
    label_col = args.label

    # 1. 加载精选因子数据
    factor_list = load_factor_list(label_col=label_col)
    if factor_list:
        print(f"[INFO] 使用精选因子清单: {len(factor_list)} 个")
    df, feature_cols = prepare_data(label_col=label_col, factor_list=factor_list)
    print(f"[INFO] 特征数: {len(feature_cols)}, 标签: {label_col}, 样本: {len(df):,}")

    # 2. 验证集内快速搜索窗口
    X_tr, y_tr, X_va, y_va = _quick_split(
        df, feature_cols, label_col,
        train_days=ROLLING["train_len"], valid_days=ROLLING["valid_len"],
        limit_filter=args.limit_filter,
    )

    # 3. Optuna 搜索
    import optuna

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(X_tr, y_tr, X_va, y_va, label_col),
                   n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"🏆 最优 trial #{best.number}: valid RankIC = {best.value:.6f}")
    print("最优超参:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # 4. 最优超参落盘（覆盖 DEFAULT_LGB_PARAMS 的默认值）
    best_params = {**DEFAULT_LGB_PARAMS, **best.params}
    # 统一命名：把 Optuna 的 feature_fraction/bagging_fraction 映射回
    # LightGBM 原生名 colsample_bytree/subsample
    if "feature_fraction" in best_params:
        best_params["colsample_bytree"] = best_params.pop("feature_fraction")
    if "bagging_fraction" in best_params:
        best_params["subsample"] = best_params.pop("bagging_fraction")
        best_params["subsample_freq"] = 1

    # 按标签分条存储：多个标签的搜索互不覆盖
    # 结构: {"label_ret_20": {...}, "label_ret_10": {...}}
    best_path = MODELS_LGB_DIR / "best_params.json"
    MODELS_LGB_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if best_path.exists():
        with open(best_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "label" in data:  # 兼容旧单条格式
                data = {data["label"]: data}
    data[label_col] = {
        "label": label_col,
        "n_trials": n_trials,
        "objective": "valid_rank_ic",
        "best_trial": int(best.number),
        "best_rank_ic": float(best.value),
        "params": best_params,
        "created_by": "optuna_search.py",
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 最优超参已保存: {best_path}")
    print(f"[INFO] 已记录的标签: {list(data.keys())}")


if __name__ == "__main__":
    sys.exit(main())
