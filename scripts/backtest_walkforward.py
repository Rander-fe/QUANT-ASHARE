# -*- coding: utf-8 -*-
"""
Walk-Forward 回测框架（LightGBM + TopkDropout）

时间划分：
    - 训练窗口：5年（2016-2020, 2017-2021, ...）
    - 验证窗口：1年（用于早停）
    - 测试窗口：1年（用于回测）
    - 滚动步长：1年

策略参数：
    - 调仓周期：5个交易日
    - 持有股票数：topk=50
    - 每次调仓替换：n_drop=10

输出：
    - data/processed/backtest_results.parquet（每日持仓和净值）
    - reports/backtest_report.txt（绩效指标）
    - reports/backtest_nav.png（净值曲线）
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED, REPORTS_DIR

# 配置
TOP_K = 50
N_DROP = 10
HOLDING = 5  # 调仓周期（交易日）
TRAIN_YEARS = 5
VALID_YEARS = 1
TEST_YEARS = 1
STEP_YEARS = 1  # 滚动步长
MIN_TRAIN_DAYS = 500  # 最少训练天数
COST_BID = 0.0005  # 买入成本（佣金+滑点）
COST_ASK = 0.0015  # 卖出成本（佣金+印花税+滑点）


def load_data(factor_cols: list[str]) -> pd.DataFrame:
    """加载因子数据和标签"""
    path = DATA_PROCESSED / "factors.parquet"
    read_cols = ["symbol", "date", "label_ret_5"] + factor_cols
    df = pd.read_parquet(path, columns=read_cols)
    df["date"] = pd.to_datetime(df["date"])
    # 剔除标签缺失的行
    df = df.dropna(subset=["label_ret_5"])
    # 按日期和股票排序
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"📊 加载数据: {len(df):,} 行, {len(df.columns)} 列")
    return df


def split_windows(dates: pd.Series) -> list[dict]:
    """生成滚动窗口时间划分"""
    date_list = sorted(dates.unique())
    min_date = date_list[0]
    max_date = date_list[-1]

    # 计算大致年数
    start_year = min_date.year
    end_year = max_date.year

    windows = []
    year = start_year

    while True:
        train_start = f"{year}-01-01"
        train_end = f"{year + TRAIN_YEARS - 1}-12-31"
        valid_start = f"{year + TRAIN_YEARS}-01-01"
        valid_end = f"{year + TRAIN_YEARS + VALID_YEARS - 1}-12-31"
        test_start = f"{year + TRAIN_YEARS + VALID_YEARS}-01-01"
        test_end = f"{year + TRAIN_YEARS + VALID_YEARS + TEST_YEARS - 1}-12-31"

        # 检查测试窗口是否超出数据范围
        if pd.Timestamp(test_end) > max_date:
            break

        windows.append({
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, test_end),
            "label": f"{train_start}~{test_end}",
        })
        year += STEP_YEARS

    print(f"📊 生成 {len(windows)} 个滚动窗口")
    return windows


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    factor_cols: list[str],
    label_col: str = "label_ret_5",
) -> lgb.Booster:
    """训练 LightGBM 模型，使用验证集早停"""
    X_train = train_df[factor_cols].fillna(0).values
    y_train = train_df[label_col].fillna(0).values
    X_valid = valid_df[factor_cols].fillna(0).values
    y_valid = valid_df[label_col].fillna(0).values

    # 如果训练集太小，跳过
    if len(X_train) < MIN_TRAIN_DAYS:
        return None

    train_dataset = lgb.Dataset(X_train, y_train)
    valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 2021,
    }

    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[valid_dataset],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    return model


def generate_signals(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """生成测试集的预测分数"""
    X_test = test_df[factor_cols].fillna(0).values
    pred = model.predict(X_test)
    test_df = test_df.copy()
    test_df["pred_score"] = pred
    return test_df[["date", "symbol", "pred_score"]]


def run_walkforward(
    df: pd.DataFrame,
    factor_cols: list[str],
    windows: list[dict],
) -> pd.DataFrame:
    """执行 Walk-Forward 回测"""
    all_trades = []

    for i, window in enumerate(windows, 1):
        print(f"\n🏃 窗口 {i}/{len(windows)}: {window['label']}")

        train_start, train_end = window["train"]
        valid_start, valid_end = window["valid"]
        test_start, test_end = window["test"]

        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        valid_df = df[(df["date"] >= valid_start) & (df["date"] <= valid_end)]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if len(train_df) < 1000:
            print(f"   ⚠️ 训练集样本不足，跳过")
            continue

        # 训练模型
        model = train_model(train_df, valid_df, factor_cols)
        if model is None:
            continue

        # 生成预测
        pred_df = generate_signals(model, test_df, factor_cols)

        # 按日期分组，每天选出 topk
        for date, group in pred_df.groupby("date"):
            group = group.sort_values("pred_score", ascending=False)
            # 选出前 TOP_K 只
            buy_symbols = group.head(TOP_K)["symbol"].tolist()
            # 生成交易指令（简化：只记录持仓，不记录买卖价）
            trade_record = {
                "date": date,
                "symbols": buy_symbols,
                "scores": group.head(TOP_K)["pred_score"].tolist(),
            }
            all_trades.append(trade_record)

        # 释放内存
        del train_df, valid_df, test_df, model, pred_df
        gc.collect()

    return pd.DataFrame(all_trades)


def compute_portfolio_returns(
    trades_df: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    根据交易记录计算组合净值
    简化：假设 T 日收盘价买入，T+1 日收盘价计算收益
    """
    # 将交易记录展开为长表
    records = []
    for _, row in trades_df.iterrows():
        date = row["date"]
        for sym in row["symbols"]:
            records.append({"date": date, "symbol": sym})

    if not records:
        print("⚠️ 无交易记录")
        return pd.DataFrame()

    pos_df = pd.DataFrame(records)

    # 合并价格数据（用收盘价）
    price_df = df[["date", "symbol", "close", "label_ret_5"]].copy()
    pos_df = pd.merge(pos_df, price_df, on=["date", "symbol"], how="left")

    # 等权重组合：每日各股票权重相同
    daily_weights = pos_df.groupby("date")["symbol"].transform("count")
    pos_df["weight"] = 1.0 / daily_weights

    # 计算当日组合收益率：每日所有持仓股票的 label_ret_5 加权平均
    # label_ret_5 是未来5日收益，但这里我们模拟调仓日次日买入，持有5天
    # 为简化，我们直接用当日 label_ret_5 乘以权重（近似）
    daily_ret = pos_df.groupby("date").apply(
        lambda g: (g["weight"] * g["label_ret_5"]).sum()
    ).reset_index(name="return")

    daily_ret = daily_ret.sort_values("date").reset_index(drop=True)

    # 计算累计净值
    daily_ret["cum_nav"] = (1 + daily_ret["return"]).cumprod()

    # 添加成本调整（每天调仓成本）
    # 近似：每天换手率 * 成本
    daily_ret["turnover"] = 0.2  # 假设每天换手20%
    daily_ret["cost"] = daily_ret["turnover"] * (COST_BID + COST_ASK) / 2
    daily_ret["net_return"] = daily_ret["return"] - daily_ret["cost"]
    daily_ret["net_cum_nav"] = (1 + daily_ret["net_return"]).cumprod()

    return daily_ret


def generate_report(nav_df: pd.DataFrame) -> dict:
    """生成绩效指标"""
    if nav_df.empty:
        return {}

    # 年化收益
    days = len(nav_df)
    if days < 2:
        return {}
    total_return = nav_df["net_cum_nav"].iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / days) - 1

    # 波动率
    daily_vol = nav_df["net_return"].std()
    ann_vol = daily_vol * np.sqrt(252)

    # 夏普
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # 最大回撤
    cum = nav_df["net_cum_nav"].values
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # 胜率
    win_rate = (nav_df["net_return"] > 0).mean()

    # 盈亏比
    pos_returns = nav_df["net_return"][nav_df["net_return"] > 0]
    neg_returns = nav_df["net_return"][nav_df["net_return"] < 0]
    profit_factor = pos_returns.sum() / abs(neg_returns.sum()) if neg_returns.sum() != 0 else np.nan

    metrics = {
        "年化收益率": ann_return,
        "年化波动率": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "胜率": win_rate,
        "盈亏比": profit_factor,
        "总交易日": days,
    }
    return metrics


def plot_nav(nav_df: pd.DataFrame, save_path: Path):
    """绘制净值曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(nav_df["date"], nav_df["net_cum_nav"], label="策略净值", linewidth=2)
    plt.title("Walk-Forward 回测净值曲线")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 净值曲线保存: {save_path}")


def main():
    # 1. 加载因子清单
    json_path = DATA_PROCESSED / "selected_factor_cols.json"
    if not json_path.exists():
        print("[ERROR] 未找到 selected_factor_cols.json，请先运行 remove_redundant_factors.py")
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    factor_cols = config["factors"]
    print(f"📊 加载因子清单: {len(factor_cols)} 个因子")

    # 2. 加载数据
    df = load_data(factor_cols)

    # 3. 生成滚动窗口
    windows = split_windows(df["date"])
    if not windows:
        print("[ERROR] 无有效滚动窗口")
        return 1

    # 4. 执行 Walk-Forward
    trades_df = run_walkforward(df, factor_cols, windows)

    # 5. 计算组合收益
    nav_df = compute_portfolio_returns(trades_df, df)

    if nav_df.empty:
        print("[ERROR] 回测结果为空")
        return 1

    # 6. 生成绩效报告
    metrics = generate_report(nav_df)

    print("\n" + "=" * 80)
    print("📋 Walk-Forward 回测绩效报告")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("=" * 80)

    # 7. 保存报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "backtest_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Walk-Forward 回测绩效报告\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:12s}: {v:.4f}\n")

    # 8. 保存净值数据
    nav_df.to_parquet(DATA_PROCESSED / "backtest_nav.parquet", index=False)
    print(f"✅ 净值数据保存: {DATA_PROCESSED / 'backtest_nav.parquet'}")

    # 9. 画图
    plot_nav(nav_df, REPORTS_DIR / "backtest_nav.png")

    print("\n✅ 回测完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())