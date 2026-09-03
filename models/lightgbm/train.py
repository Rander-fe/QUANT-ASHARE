# -*- coding: utf-8 -*-
"""LightGBM 滚动重训（Rolling Retrain）选股模型。

流程（借鉴 Qlib 官方 RR 基准 examples/benchmarks_dynamic/baseline）：
    1. 加载预处理产物（features.parquet：去极值+行业/市值中性化+标准化；
       labels.parquet：原值标签），剔除指数/B 股
    2. 按滚动窗口 [train | valid | test] 划分（train/valid 严格在 test 之前）
    3. 每步用 train 训练 LightGBM（valid 早停），预测 test 段
    4. 汇总全部滚动预测 -> data/processed/predictions/lgb_pred_{label}.parquet
    5. 模型逐步入 models/lightgbm/，MLflow 记录到 mlruns/
    6. 默认只生成到验证期末的预测；测试期预测必须显式确认

用法：
    python models/lightgbm/train.py                     # 全量滚动
    python models/lightgbm/train.py --quick             # 冒烟（限 3 步）
    python models/lightgbm/train.py --label label_ret_10 --step 10
    python models/lightgbm/train.py --limit-filter lock # 只剔一字板样本
    python models/lightgbm/train.py --eval-segments valid   # 只评验证集（标签选择阶段，铁律）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    DATA_PROCESSED,
    LGB_REPORTS_DIR,
    MODELS_LGB_DIR,
    PREDICTIONS_DIR,
    VALID_PERIOD,
    TEST_PERIOD,
)
from models.lightgbm.config import (
    DEFAULT_LGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    LABEL_COL,
    NUM_BOOST_ROUND,
    ROLLING,
)
from models.lightgbm.data import (
    build_rolling_windows,
    label_horizon,
    load_best_params,
    load_factor_list,
    prepare_data,
)
from models.lightgbm.evaluate import evaluate_predictions, save_daily_ic


# 时间划分（铁律）：验证集仅用于模型选择；测试集只评估一次
VALID_START, VALID_END = VALID_PERIOD
TEST_START, TEST_END = TEST_PERIOD


def _segment_of(date: pd.Timestamp) -> str:
    """按铁律把日期归入 valid / test / other。"""
    if pd.Timestamp(VALID_START) <= date <= pd.Timestamp(VALID_END):
        return "valid"
    if pd.Timestamp(TEST_START) <= date <= pd.Timestamp(TEST_END):
        return "test"
    return "other"


def _limit_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    """返回可训练样本掩码（剔除封板日，标签收益不可实现）。

    mode 取值：
      all  : 剔除涨停 + 跌停日样本（默认，业界主流：
             涨停日收盘买不进、标签用涨停收盘价会高估收益；
             跌停日流动性枯竭、收益不可靠）
      lock : 只剔除一字板（lock_limit_up 买不进 / lock_limit_down 卖不出）
      none : 不过滤
    列缺失时安全回退为全保留。
    """
    if mode == "none" or df.empty:
        return pd.Series(True, index=df.index)

    if mode == "all":
        cols = ["limit_up", "limit_down"]
        if not set(cols) <= set(df.columns):
            return pd.Series(True, index=df.index)
        flags = df[cols].fillna(False).any(axis=1)
    elif mode == "lock":
        cols = ["lock_limit_up", "lock_limit_down"]
        if not set(cols) <= set(df.columns):
            return pd.Series(True, index=df.index)
        flags = df[cols].fillna(False).any(axis=1)
    else:
        return pd.Series(True, index=df.index)
    return ~flags


def rolling_train(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                  rolling: dict | None = None, max_steps: int | None = None,
                  quick: bool = False,
                  limit_filter: str = "all",
                  params_override: dict | None = None) -> tuple[pd.DataFrame, list]:
    """执行滚动重训，返回 (预测长表, 每步模型路径列表)。

    params_override : 覆盖默认超参（如 Optuna 搜索得到的最优超参）。
    """
    if rolling is None:
        rolling = ROLLING
    step = int(rolling["step"])

    dates = np.sort(df["date"].unique())
    purge_days = label_horizon(label_col)
    windows = build_rolling_windows(dates, rolling, purge_days=purge_days)
    if quick:
        windows = windows[-3:] if len(windows) >= 3 else windows
        print(f"[INFO] quick 模式：仅执行最后 {len(windows)} 个滚动步")
    if max_steps:
        windows = windows[-max_steps:]
        print(f"[INFO] 限制滚动步数：{len(windows)}")

    params = {**DEFAULT_LGB_PARAMS, "num_threads": 8}
    if params_override:
        params = {**params, **params_override, "num_threads": 8}
    print(f"[INFO] 滚动窗口总数: {len(windows)}（每步 test={step} 交易日，"
          f"边界隔离={purge_days} 交易日）")

    pred_frames: list[pd.DataFrame] = []
    model_paths: list[str] = []

    for i, (tr_idx, va_idx, te_idx) in enumerate(windows):
        tr_dates = pd.to_datetime(dates[tr_idx])
        va_dates = pd.to_datetime(dates[va_idx])
        te_dates = pd.to_datetime(dates[te_idx])

        train_df = df[df["date"].isin(tr_dates)]
        valid_df = df[df["date"].isin(va_dates)]
        test_df = df[df["date"].isin(te_dates)]

        # 特征列已在 prepare_data 统一为 float32，此处切片即视图，无需再 astype
        # （astype 会对每步 3 份全量切片做深拷贝，89 步 × 3 次会白费大量内存）
        X_tr = train_df[feature_cols]
        y_tr = train_df[label_col]
        X_va = valid_df[feature_cols]
        y_va = valid_df[label_col]
        X_te = test_df[feature_cols]

        # 丢弃标签缺失样本（标签列首尾窗口）+ 封板日样本（标签收益不可实现）
        # _limit_mask 每步只算一次（train/valid 各一次），避免重复构造布尔数组
        limit_tr = _limit_mask(train_df, limit_filter).values
        limit_va = _limit_mask(valid_df, limit_filter).values
        keep_tr = y_tr.notna() & np.isfinite(y_tr) & limit_tr
        keep_va = y_va.notna() & np.isfinite(y_va) & limit_va
        n_filt_tr = int((~limit_tr).sum())
        n_filt_va = int((~limit_va).sum())
        if n_filt_tr or n_filt_va:
            print(f"[INFO] 封板日过滤: train 剔除 {n_filt_tr:,} 行, valid 剔除 {n_filt_va:,} 行")
        dtrain = lgb.Dataset(X_tr[keep_tr], label=y_tr[keep_tr])
        dvalid = lgb.Dataset(X_va[keep_va], label=y_va[keep_va], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=0)],
        )

        # 预测 test 段
        pred = model.predict(X_te, num_iteration=model.best_iteration or NUM_BOOST_ROUND)
        step_pred = test_df[["symbol", "date", label_col]].copy()
        step_pred["pred"] = pred
        step_pred["segment"] = step_pred["date"].map(_segment_of)
        # 保存每个分数对应的训练边界，便于审计 OOS/OOF 和后续残差挖掘。
        step_pred["model_train_start"] = tr_dates[0]
        step_pred["model_train_end"] = tr_dates[-1]
        step_pred["model_valid_start"] = va_dates[0]
        step_pred["model_valid_end"] = va_dates[-1]
        pred_frames.append(step_pred)

        # 落盘模型（lightgbm 原生格式）
        step_dir = MODELS_LGB_DIR / label_col / f"step_{i:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        model_path = step_dir / "lgb_model.txt"
        model.save_model(str(model_path))
        model_paths.append(str(model_path))

        if (i + 1) % 5 == 0 or i == len(windows) - 1:
            print(f"[INFO] 滚动步 {i + 1}/{len(windows)}: "
                  f"train {tr_dates[0].date()}~{tr_dates[-1].date()} | "
                  f"valid {va_dates[0].date()}~{va_dates[-1].date()} | "
                  f"test {te_dates[0].date()}~{te_dates[-1].date()}")

    pred_df = pd.concat(pred_frames, ignore_index=True)
    return pred_df, model_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGBM 滚动重训选股")
    parser.add_argument("--label", default=LABEL_COL, help="标签列")
    parser.add_argument("--step", type=int, default=ROLLING["step"], help="滚动步长（交易日）")
    parser.add_argument("--train-len", type=int, default=ROLLING["train_len"], help="训练窗口（交易日）")
    parser.add_argument("--valid-len", type=int, default=ROLLING["valid_len"], help="验证窗口（交易日）")
    parser.add_argument("--max-steps", type=int, default=None, help="限制滚动步数（调试用）")
    parser.add_argument("--quick", action="store_true", help="冒烟模式：仅最后 3 步")
    parser.add_argument(
        "--limit-filter", choices=("all", "lock", "none"), default="all",
        help="训练/验证样本过滤封板日: all=剔涨停+跌停(默认), lock=仅剔一字板, none=不过滤",
    )
    parser.add_argument(
        "--eval-segments", default="valid",
        help="评估区段（默认仅 valid；测试集需额外显式确认）",
    )
    parser.add_argument(
        "--prediction-through", default="valid", choices=("valid", "test"),
        help="生成预测的最远区段；默认 valid，防止训练流程自动触碰测试期",
    )
    parser.add_argument(
        "--confirm-final-test", action="store_true",
        help="确认执行一次性测试集评估；没有此参数时拒绝评估 test",
    )
    args = parser.parse_args()

    eval_segments = [s.strip() for s in args.eval_segments.split(",") if s.strip()]
    if "test" in eval_segments and not args.confirm_final_test:
        parser.error("测试集仅允许最终评估；请在方案冻结后显式传入 --confirm-final-test")
    if args.prediction_through == "test" and not args.confirm_final_test:
        parser.error("生成测试期预测也属于触碰测试集；请显式传入 --confirm-final-test")

    label_col = args.label
    rolling = {"step": args.step, "train_len": args.train_len, "valid_len": args.valid_len}

    # 1. 加载与预处理
    print("=" * 60)
    print("LightGBM 滚动重训")
    print("=" * 60)
    factor_list = load_factor_list(label_col=label_col)
    if factor_list:
        print(f"[INFO] 使用精选因子清单: {len(factor_list)} 个（分层回测通过）")
    df, feature_cols = prepare_data(label_col=label_col, factor_list=factor_list)
    prediction_end = pd.Timestamp(VALID_END if args.prediction_through == "valid" else TEST_END)
    df = df[df["date"] <= prediction_end].copy()
    print(f"[INFO] 预测数据截止: {prediction_end.date()}（{args.prediction_through}）")
    print(f"[INFO] 特征数: {len(feature_cols)}, 标签: {label_col}, "
          f"样本: {len(df):,} 行")

    # 最优超参（Optuna）覆盖默认（按当前标签取）
    best_params = load_best_params(label_col=label_col)

    # 2. 滚动训练
    pred_df, model_paths = rolling_train(df, feature_cols, label_col,
                                         rolling=rolling, max_steps=args.max_steps,
                                         quick=args.quick,
                                         limit_filter=args.limit_filter,
                                         params_override=best_params)
    print(f"[INFO] 总预测行数: {len(pred_df):,}")

    # 3. 预测落盘（按标签分文件，避免三标签互相覆盖，供对比分析）
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PREDICTIONS_DIR / f"lgb_pred_{label_col}.parquet"
    # 评分产物不携带真实未来收益，避免测试集“答案”随预测一起泄露给下游。
    prediction_columns = [
        "symbol", "date", "pred", "segment", "model_train_start", "model_train_end",
        "model_valid_start", "model_valid_end",
    ]
    pred_df[prediction_columns].to_parquet(pred_path, index=False)
    print(f"[OK] 预测已保存: {pred_path}")

    # 4. 分铁律区段评估（区段由 --eval-segments 控制）
    LGB_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n评估（按铁律区段）:")
    for seg in eval_segments:
        if seg not in ("valid", "test"):
            print(f"  [WARN] 未知区段 {seg}，跳过")
            continue
        seg_df = pred_df[pred_df["segment"] == seg]
        if seg_df.empty:
            print(f"  [{seg}] 无预测样本，跳过")
            continue
        metrics = evaluate_predictions(seg_df, "pred", label_col)
        tag = "（最终评估，仅此一次）" if seg == "test" else "（模型选择用）"
        print(f"  [{seg}] {seg_df['date'].min().date()}~{seg_df['date'].max().date()} {tag}")
        print(f"      IC={metrics['ic_mean']}, ICIR={metrics['icir']}, "
              f"RankIC={metrics['rank_ic_mean']}, RankICIR={metrics['rank_icir']}, "
              f"TOP组收益={metrics['top_group_mean']}, 多空={metrics['long_short_spread']}")

        # 逐日 IC 落盘（reports/lightgbm/）
        from models.lightgbm.evaluate import calc_ic
        daily = calc_ic(seg_df, "pred", label_col)
        save_daily_ic(daily, LGB_REPORTS_DIR / label_col / f"daily_ic_{seg}.parquet")

        # MLflow 记录（实验级指标）
        try:
            import mlflow

            tracking_dir = Path(__file__).resolve().parent.parent.parent / "mlruns"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
            with mlflow.start_run(run_name=f"lgb_rolling_{seg}"):
                mlflow.log_params({**rolling, "label": label_col,
                                   "num_features": len(feature_cols),
                                   "num_boost_round": NUM_BOOST_ROUND,
                                   "early_stopping_rounds": EARLY_STOPPING_ROUNDS})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                mlflow.log_artifact(str(pred_path))
        except Exception as e:  # pragma: no cover - MLflow 不可用时降级
            print(f"  [WARN] MLflow 记录失败: {e}")

    # 5. 最终模型信息
    print(f"\n[OK] 模型已保存: {MODELS_LGB_DIR / label_col}（共 {len(model_paths)} 个滚动模型）")
    print(f"[OK] 评估报告: {LGB_REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
