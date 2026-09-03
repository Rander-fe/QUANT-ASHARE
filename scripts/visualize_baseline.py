# -*- coding: utf-8 -*-
"""
Visualize the LightGBM baseline backtest results from mlruns artifacts.

Loads the portfolio report / IC series / positions saved by
`scripts/lgb_baseline.py` (qlib PortAnaRecord + SigAnaRecord artifacts) and
renders publication-style PNG figures.

Run:
    C:/Users/haoran/miniconda3/envs/rqalpha/python.exe scripts/visualize_baseline.py [run_dir]

`run_dir` is optional; by default the most recently modified run under
`mlruns/<experiment_id>/` is used.
"""
import glob
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Chinese-capable fonts on Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

MLRUNS_ROOT = "c:/QUANT-ASHARE/mlruns"
OUT_DIR = "c:/QUANT-ASHARE/reports"


def _pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_latest_run(experiment_id=None):
    """Return the path of the most recently modified run directory."""
    if experiment_id is None:
        exp_dirs = [
            d
            for d in glob.glob(os.path.join(MLRUNS_ROOT, "*"))
            if os.path.isdir(d) and os.path.basename(d) != "0"
        ]
        if not exp_dirs:
            raise FileNotFoundError("no experiment dirs under mlruns/")
        experiment_id = os.path.basename(max(exp_dirs, key=os.path.getmtime))
    run_dirs = [
        d
        for d in glob.glob(os.path.join(MLRUNS_ROOT, experiment_id, "*"))
        if os.path.isdir(d)
    ]
    if not run_dirs:
        raise FileNotFoundError(f"no run dirs under experiment {experiment_id}")
    return max(run_dirs, key=os.path.getmtime)


def load_artifacts(run_dir):
    base = os.path.join(run_dir, "artifacts")
    report = _pickle_load(os.path.join(base, "portfolio_analysis", "report_normal_1day.pkl"))
    ic = _pickle_load(os.path.join(base, "sig_analysis", "ic.pkl"))
    ric = _pickle_load(os.path.join(base, "sig_analysis", "ric.pkl"))
    positions = _pickle_load(
        os.path.join(base, "portfolio_analysis", "positions_normal_1day.pkl")
    )
    port_ana = _pickle_load(
        os.path.join(base, "portfolio_analysis", "port_analysis_1day.pkl")
    )
    return report, ic, ric, positions, port_ana


def plot_net_value_and_drawdown(report, out):
    cum_port = (1 + report["return"]).cumprod()
    cum_bench = (1 + report["bench"]).cumprod()
    # drawdown of portfolio & benchmark
    dd_port = cum_port / cum_port.cummax() - 1
    dd_bench = cum_bench / cum_bench.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(cum_port.index, cum_port, label="策略组合", lw=1.6, color="#d62728")
    axes[0].plot(cum_bench.index, cum_bench, label="基准 沪深300", lw=1.6, color="#1f77b4")
    axes[0].set_ylabel("累计净值")
    axes[0].set_title("LightGBM 基线：累计净值（测试集 2025-01 ~ 2026-08）")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(dd_port.index, dd_port * 100, 0, color="#d62728", alpha=0.35, label="策略回撤")
    axes[1].fill_between(dd_bench.index, dd_bench * 100, 0, color="#1f77b4", alpha=0.25, label="基准回撤")
    axes[1].set_ylabel("回撤 (%)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_ic(ic, ric, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, s, name, color in [
        (axes[0], ic, "IC", "#1f77b4"),
        (axes[1], ric, "Rank IC", "#2ca02c"),
    ]:
        s = s.dropna()
        ax.bar(s.index, s, width=0.8, color=color, alpha=0.5, label=name)
        ax.plot(s.index, s.rolling(20).mean(), color=color, lw=1.8, label=f"{name} 20日均值")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(name)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[0].set_title("IC / Rank IC 时间序列（测试集）")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_cum_ic(ic, ric, out):
    ic = ic.dropna()
    ric = ric.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ic.index, ic.cumsum(), color="#1f77b4", lw=1.6)
    axes[0].set_ylabel("累计 IC")
    axes[0].set_title("累计 IC / 累计 Rank IC（测试集）")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ric.index, ric.cumsum(), color="#2ca02c", lw=1.6)
    axes[1].set_ylabel("累计 Rank IC")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def plot_positions_and_turnover(report, positions, out):
    # number of held stocks per day
    pos_counts = pd.Series(
        {ts: (len(v) if hasattr(v, "__len__") else 0) for ts, v in positions.items()}
    ).sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(pos_counts.index, pos_counts, color="#9467bd", lw=1.4)
    axes[0].set_ylabel("持仓股票数")
    axes[0].set_title("持仓数量与换手率（测试集）")
    axes[0].grid(alpha=0.3)

    axes[1].plot(report.index, report["turnover"] * 100, color="#ff7f0e", lw=1.0)
    axes[1].set_ylabel("换手率 (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def print_summary(report, ic, ric, port_ana):
    print("\n===== 回测摘要 =====")
    cum_port = (1 + report["return"]).cumprod()
    cum_bench = (1 + report["bench"]).cumprod()
    n_days = len(report)
    ann_port = cum_port.iloc[-1] ** (252 / n_days) - 1
    ann_bench = cum_bench.iloc[-1] ** (252 / n_days) - 1
    print(f"回测区间: {report.index[0].date()} ~ {report.index[-1].date()}  ({n_days} 个交易日)")
    print(f"策略累计净值: {cum_port.iloc[-1]:.4f}   年化: {ann_port:.2%}")
    print(f"基准累计净值: {cum_bench.iloc[-1]:.4f}   年化: {ann_bench:.2%}")
    print(f"策略最大回撤: {(cum_port / cum_port.cummax() - 1).min():.2%}")
    print(f"基准最大回撤: {(cum_bench / cum_bench.cummax() - 1).min():.2%}")
    ic_clean = ic.dropna()
    ric_clean = ric.dropna()
    print(f"IC   均值 {ic_clean.mean():.4f}   ICIR {ic_clean.mean() / (ic_clean.std() + 1e-12):.4f}")
    print(f"RankIC 均值 {ric_clean.mean():.4f}  RankICIR {ric_clean.mean() / (ric_clean.std() + 1e-12):.4f}")
    print("\n组合分析(risk 指标):")
    print(port_ana.to_string())


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run dir: {run_dir}")
    report, ic, ric, positions, port_ana = load_artifacts(run_dir)

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_net_value_and_drawdown(report, os.path.join(OUT_DIR, "01_net_value_drawdown.png"))
    plot_ic(ic, ric, os.path.join(OUT_DIR, "02_ic_ric.png"))
    plot_cum_ic(ic, ric, os.path.join(OUT_DIR, "03_cum_ic_ric.png"))
    plot_positions_and_turnover(report, positions, os.path.join(OUT_DIR, "04_positions_turnover.png"))

    print_summary(report, ic, ric, port_ana)
    print(f"\n[OK] 图表已输出到 {OUT_DIR}")


if __name__ == "__main__":
    main()
