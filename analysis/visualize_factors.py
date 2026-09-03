# -*- coding: utf-8 -*-
"""
因子可视化分析（基于 Seaborn + Matplotlib）

功能：
    1. Top N 因子 IC / ICIR 条形图
    2. 指定因子 IC 时间序列（稳定性）
    3. 因子相关性热力图（识别冗余）
    4. IC vs ICIR 散点图（四象限分类）

输入：
    - data/processed/factor_evaluation.parquet（IC/IR 汇总表）
    - data/processed/daily_rankic.parquet（日频 IC 序列）
    - data/processed/factors.parquet（原始因子数据，用于计算相关性）

输出：
    - data/processed/figures/ 目录下保存 PNG 图片
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_PROCESSED

# 设置 Seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def plot_top_ic_bar(summary: pd.DataFrame, top_n: int = 30, save_path: Path = None):
    """绘制 Top N 因子的 IC 均值条形图（附加 ICIR 作为颜色映射）"""
    top = summary.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.3)))
    
    # 使用 ICIR 作为颜色映射
    norm = plt.Normalize(top["icir"].min(), top["icir"].max())
    colors = plt.cm.RdYlGn(norm(top["icir"]))
    
    bars = ax.barh(top["factor"], top["ic_mean"], color=colors, edgecolor="gray", linewidth=0.5)
    
    # 标注 ICIR
    for i, (idx, row) in enumerate(top.iterrows()):
        ax.text(
            row["ic_mean"] + 0.002,
            i,
            f'ICIR={row["icir"]:.2f}',
            va="center",
            fontsize=9,
            color="gray",
        )
    
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(0.02, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="IC=0.02 (可用阈值)")
    ax.axvline(0.05, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="IC=0.05 (优秀阈值)")
    
    ax.set_xlabel("平均 RankIC")
    ax.set_title(f"Top {top_n} 因子平均 IC 及 ICIR（颜色深浅表示 ICIR 高低）")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 已保存 Top IC 条形图: {save_path}")


def plot_ic_timeseries(
    daily_ic: pd.DataFrame,
    factor_names: list = None,
    top_n: int = 5,
    save_path: Path = None,
):
    """绘制指定因子的 IC 时间序列（检验稳定性）"""
    if factor_names is None:
        # 默认取 IC 均值最高的前 N 个因子
        factor_names = daily_ic.mean().sort_values(ascending=False).head(top_n).index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for factor in factor_names:
        if factor in daily_ic.columns:
            # 滚动 60 日均值平滑
            ic_series = daily_ic[factor].dropna()
            ic_smooth = ic_series.rolling(60).mean()
            ax.plot(ic_series.index, ic_smooth, label=factor, linewidth=1.5)
    
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(0.02, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="IC=0.02")
    ax.axhline(-0.02, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel("日期")
    ax.set_ylabel("滚动 60 日平均 RankIC")
    ax.set_title(f"Top {len(factor_names)} 因子 IC 时间序列（滚动平均）")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 已保存 IC 时间序列图: {save_path}")


def plot_correlation_heatmap(
    df_factors: pd.DataFrame,
    summary: pd.DataFrame,
    top_n: int = 30,
    save_path: Path = None,
):
    """绘制 Top N 因子的相关性热力图（识别冗余对）"""
    top_factors = summary.head(top_n)["factor"].tolist()
    # 只取存在于 df_factors 中的列
    available = [f for f in top_factors if f in df_factors.columns]
    
    if len(available) < 2:
        print("⚠️ 可用因子不足 2 个，跳过相关性热力图")
        return
    
    corr = df_factors[available].corr(method="spearman")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 创建 mask 只显示上三角（避免冗余）
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman 相关系数"},
        ax=ax,
    )
    
    ax.set_title(f"Top {len(available)} 因子相关性热力图（|corr| > 0.7 标记为冗余）")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 已保存相关性热力图: {save_path}")


def plot_ic_vs_icir(summary: pd.DataFrame, save_path: Path = None):
    """绘制 IC vs ICIR 散点图（四象限分类）"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 分类颜色：高 IC & 高 ICIR 为绿色
    colors = []
    for _, row in summary.iterrows():
        if row["ic_mean"] > 0.02 and row["icir"] > 0.05:
            colors.append("green")
        elif row["ic_mean"] > 0.02:
            colors.append("orange")
        elif row["icir"] > 0.05:
            colors.append("blue")
        else:
            colors.append("gray")
    
    scatter = ax.scatter(
        summary["ic_mean"],
        summary["icir"],
        c=colors,
        alpha=0.7,
        s=60,
        edgecolors="white",
        linewidth=0.5,
    )
    
    # 标注 Top 5 因子
    top5 = summary.head(5)
    for _, row in top5.iterrows():
        ax.annotate(
            row["factor"],
            (row["ic_mean"], row["icir"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    
    ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="ICIR=0.05 (可用下限)")
    ax.axvline(0.02, color="red", linestyle="--", alpha=0.5, label="IC=0.02 (可用下限)")
    
    ax.set_xlabel("平均 RankIC")
    ax.set_ylabel("ICIR (IC 均值 / IC 标准差)")
    ax.set_title("因子质量四象限图（绿色：高 IC + 高 ICIR，优先保留）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 已保存 IC vs ICIR 散点图: {save_path}")


def main():
    # 1. 创建图片输出目录
    fig_dir = DATA_PROCESSED / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 图片保存目录: {fig_dir}")

    # 2. 加载评估结果
    summary_path = DATA_PROCESSED / "factor_evaluation.parquet"
    daily_ic_path = DATA_PROCESSED / "daily_rankic.parquet"
    factors_path = DATA_PROCESSED / "factors.parquet"

    if not summary_path.exists():
        print(f"[ERROR] 未找到 {summary_path}，请先运行 analysis/factor_evaluation.py")
        return 1

    summary = pd.read_parquet(summary_path)
    daily_ic = pd.read_parquet(daily_ic_path) if daily_ic_path.exists() else None
    df_factors = pd.read_parquet(factors_path) if factors_path.exists() else None

    print(f"📊 加载汇总数据: {len(summary)} 个因子")

    # 3. 生成各类图表
    # 3.1 Top 30 IC 条形图
    plot_top_ic_bar(
        summary,
        top_n=30,
        save_path=fig_dir / "top30_ic_bar.png",
    )

    # 3.2 IC 时间序列（Top 5 因子）
    if daily_ic is not None and not daily_ic.empty:
        plot_ic_timeseries(
            daily_ic,
            top_n=5,
            save_path=fig_dir / "top5_ic_timeseries.png",
        )

    # 3.3 相关性热力图（Top 30 因子）
    if df_factors is not None and not df_factors.empty:
        plot_correlation_heatmap(
            df_factors,
            summary,
            top_n=30,
            save_path=fig_dir / "top30_correlation.png",
        )

    # 3.4 IC vs ICIR 散点图
    plot_ic_vs_icir(
        summary,
        save_path=fig_dir / "ic_vs_icir_scatter.png",
    )

    print("\n✅ 所有可视化图表生成完成！")
    print(f"   请查看目录: {fig_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())