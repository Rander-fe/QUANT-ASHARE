# -*- coding: utf-8 -*-
"""
因子分层回测（5分组，调仓周期5日）
=====
优化版：按日期预建截面字典，所有因子共享，避免重复全表过滤。

性能提升：
    - 原版：每个因子重复全表布尔过滤（58因子 × 350调仓日 × 650万行）
    - 优化后：按 row group 流式读取 features.parquet + labels.parquet，
      一次构建截面字典 {date: DataFrame}，因子查询 O(1)

口径（与 evaluate_factors.py 保持一致，铁律）：
    - 特征用预处理后 features.parquet（已行业/市值中性化+标准化）
    - 标签用 labels.parquet 的 label_ret_5
    - 调仓网格在全区间统一建立（all_dates[::5]），再按训练/验证期过滤，
      避免分段网格在衔接处相位断裂
    - 多空收益 G5-G1 按训练集 IC 符号对齐：负向因子（如 LN_TURNOVER）
      同样能正确评估，不因方向而误杀
    - 测试集禁止使用

功能：
  1. 流式读取特征+标签 -> 截面字典 {date: DataFrame(symbol as index)}
  2. 对每个因子做 5 分组分层回测（训练集）
  3. 计算指标：单调性(|corr|)、多空收益(IC对齐)、年化/夏普/回撤、换手率
  4. 验证集单独报告（只观察不参与筛选）
  5. 落盘：factor_quantile_train.parquet + factor_quantile_valid.parquet + 报告txt
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROCESSED,
    FEATURES_FILE,
    LABELS_FILE,
    REPORTS_DIR,
    TRAIN_PERIOD,
    VALID_PERIOD,
)

# 配置
HOLDING = 5  # 调仓周期（交易日），与 label_ret_5 匹配
N_GROUPS = 5
LABEL_COL = "label_ret_5"
# features.parquet 中的元数据列（非因子），读取时排除
META_COLS = {
    "symbol",
    "date",
    "industry",
    "limit_up",
    "limit_down",
    "lock_limit_up",
    "lock_limit_down",
    "log_mv",
}


def load_selected_factors(suffix: str = "") -> list[str]:
    """加载筛选后的因子清单（ICIR 粗筛 + 相关性剔除后的因子）"""
    name = f"selected_factor_cols_{suffix}.json" if suffix else "selected_factor_cols.json"
    path = DATA_PROCESSED / name
    if not path.exists():
        print(f"[ERROR] {path} 不存在，请先运行 remove_redundant_factors.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["factors"]


def build_cross_section_dict(
    factor_cols: list[str],
) -> tuple[dict, list]:
    """
    流式读取 features.parquet（中性化特征）+ labels.parquet（标签），
    按日期分组建成截面字典 {date: DataFrame(index=symbol, columns=因子+标签)}。

    利用 row group 的 date 列统计信息跳过无因子数据的早期块，
    大幅减少 I/O。返回 (cs_dict, all_dates)。
    """
    features_path = DATA_PROCESSED / FEATURES_FILE
    labels_path = DATA_PROCESSED / LABELS_FILE
    read_cols = ["symbol", "date"] + factor_cols
    print(f"📊 流式读取特征（{len(read_cols)} 列，{features_path.name}）...")

    # 只读需要的标签列
    labels = pd.read_parquet(labels_path, columns=["symbol", "date", LABEL_COL])
    labels["date"] = pd.to_datetime(labels["date"])
    labels = labels.dropna(subset=[LABEL_COL])
    print(f"   ✅ 标签表: {len(labels):,} 行")

    # 按 row group 分块读取特征，与标签合并
    cs_dict: dict = {}
    pf = pq.ParquetFile(features_path)
    n_rg = pf.metadata.num_row_groups
    for i in range(n_rg):
        # 跳过无标签覆盖的块（标签只到 2026-08 左右，但特征到 2026-08-13，
        # 早期 2016-01 之前无数据，这里按行数直接读取，简化处理）
        chunk = pf.read_row_group(i, columns=read_cols).to_pandas()
        chunk["date"] = pd.to_datetime(chunk["date"])
        merged = chunk.merge(labels, on=["symbol", "date"], how="inner")
        del chunk
        if merged.empty:
            continue
        for date, group in merged.groupby("date", sort=True):
            group = group.set_index("symbol")
            group = group[factor_cols + [LABEL_COL]]
            cs_dict[date] = group
        del merged
    del pf

    print(f"   ✅ 建成截面字典: {len(cs_dict)} 个交易日")
    all_dates = sorted(cs_dict.keys())
    return cs_dict, all_dates


def get_rebalance_days(dates: list, holding: int = HOLDING) -> list:
    """每隔 holding 天取一个交易日作为调仓日"""
    return dates[::holding]


def compute_quantile_stats(
    group_means: list,  # 每个调仓日的组均收益，形状 (n_days, n_groups)
    ic_sign: float = 1.0,  # 训练集 IC 符号，用于对齐多空方向
) -> dict:
    """计算分层统计指标。多空收益按 ic_sign 对齐：负 IC 因子翻转 G5-G1。"""
    arr = np.array(group_means)  # (n_days, n_groups)
    n_days, n_groups = arr.shape
    # 各组的平均收益（时间序列平均）
    group_avg = np.nanmean(arr, axis=0)  # (n_groups,)

    # 单调性：Spearman(组号, 组均收益)。负向因子为负值，评估用 |corr|
    monotonic_corr, _ = stats.spearmanr(np.arange(n_groups), group_avg)

    # 多空收益（原始方向：G5 - G1），按 IC 符号对齐
    spread_raw = arr[:, -1] - arr[:, 0]  # (n_days,)
    spread = ic_sign * spread_raw

    # 计算年化收益、夏普、最大回撤（基于累计净值）
    # 注意：spread 是持有期收益（每 HOLDING 天一个），年化需按实际交易日折算
    cum_net = np.cumprod(1 + spread)
    total_periods = len(spread)
    total_trading_days = total_periods * HOLDING
    if total_trading_days > 0:
        ann_return = cum_net[-1] ** (252 / total_trading_days) - 1
    else:
        ann_return = np.nan
    # 夏普（假设无风险利率0）：持有期收益年化 = std * sqrt(252/HOLDING)
    ann_vol = np.std(spread) * np.sqrt(252 / HOLDING)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    # 最大回撤
    peak = np.maximum.accumulate(cum_net)
    drawdown = (cum_net - peak) / peak
    max_dd = drawdown.min()

    return {
        "group_avg": group_avg,
        "monotonic_corr": monotonic_corr,
        "spread_mean": float(np.mean(spread)),
        "spread_std": float(np.std(spread)),
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cum_net": cum_net,
        "ic_sign": ic_sign,
    }


def compute_turnover(
    top_members: list[list[str]],  # 每个调仓日的Top组成员symbol列表
    bottom_members: list[list[str]],
) -> float:
    """计算相邻调仓日 Top/Bottom 组的平均换手率（1 - Jaccard 重合度）"""
    if len(top_members) < 2:
        return np.nan
    turnovers = []
    for i in range(len(top_members) - 1):
        # Top组换手率
        set1, set2 = set(top_members[i]), set(top_members[i + 1])
        intersect = len(set1 & set2)
        union = len(set1 | set2)
        if union > 0:
            top_turn = 1 - intersect / union
        else:
            top_turn = np.nan
        # Bottom组换手率
        set1_b, set2_b = set(bottom_members[i]), set(bottom_members[i + 1])
        intersect_b = len(set1_b & set2_b)
        union_b = len(set1_b | set2_b)
        if union_b > 0:
            bottom_turn = 1 - intersect_b / union_b
        else:
            bottom_turn = np.nan
        turnovers.append((top_turn + bottom_turn) / 2)
    return np.nanmean(turnovers)


def run_quantile_for_factor(
    cs_dict: dict,
    rebalance_days: list,
    factor_name: str,
    ic_sign: float = 1.0,
) -> dict:
    """对单个因子运行分层回测（基于截面字典），返回统计结果 dict。"""
    group_means = []  # 每个调仓日的组均收益
    top_members = []
    bottom_members = []

    for r_date in rebalance_days:
        day_data = cs_dict.get(r_date)
        if day_data is None or len(day_data) < N_GROUPS:
            continue
        # 剔除因子缺失的股票
        valid = day_data[factor_name].notna() & day_data[LABEL_COL].notna()
        if valid.sum() < N_GROUPS:
            continue
        sub = day_data.loc[valid, [factor_name, LABEL_COL]]

        # 按因子值排序，等分 5 组（rank 后 qcut：rank 无重复，qcut 不会报错）
        ranked = sub[factor_name].rank(method="first")
        group_assign = pd.qcut(ranked, q=N_GROUPS, labels=False)
        group_assign = group_assign.astype(int)

        # 组均收益
        group_ret = sub[LABEL_COL].groupby(group_assign).mean()
        group_ret = group_ret.reindex(range(N_GROUPS), fill_value=np.nan)
        group_means.append(group_ret.values)

        # 记录Top/Bottom组成员（用于换手率）
        top_members.append(group_assign[group_assign == N_GROUPS - 1].index.tolist())
        bottom_members.append(group_assign[group_assign == 0].index.tolist())

    if len(group_means) == 0:
        return {}

    stats_dict = compute_quantile_stats(group_means, ic_sign=ic_sign)
    stats_dict["turnover"] = compute_turnover(top_members, bottom_members)
    # 累计净值单独存放（供画图），不写入汇总行
    stats_dict["cum_net"] = stats_dict["cum_net"].tolist()
    return stats_dict


def load_ic_signs(prefix: str = "") -> dict:
    """从训练集 ICIR 评估结果读取每个因子的 IC 均值符号（用于对齐多空方向）"""
    stem = f"{prefix}_" if prefix else ""
    ev_path = DATA_PROCESSED / f"{stem}factor_evaluation.parquet"
    if not ev_path.exists():
        print("[WARN] factor_evaluation.parquet 不存在，默认全部按正向处理")
        return {}
    ev = pd.read_parquet(ev_path, columns=["factor", "ic_mean"])
    return {row.factor: np.sign(row.ic_mean) for row in ev.itertuples()}


def main():
    global HOLDING, LABEL_COL
    parser = argparse.ArgumentParser(description="按目标标签进行因子分层回测")
    parser.add_argument("--label", default="label_ret_5",
                        choices=("label_ret_5", "label_ret_10", "label_ret_20"))
    parser.add_argument("--suffix", default=None,
                        help="输入输出后缀，默认由标签生成，如 ret20")
    args = parser.parse_args()
    LABEL_COL = args.label
    HOLDING = int(LABEL_COL.rsplit("_", 1)[1])
    suffix = args.suffix or f"ret{HOLDING}"

    # 1. 加载因子清单
    factor_cols = load_selected_factors(suffix)
    print(f"📊 待检验因子数: {len(factor_cols)}")

    # 2. 流式建截面字典
    cs_dict, all_dates = build_cross_section_dict(factor_cols)

    # 3. 调仓网格全局统一：全区间 [::HOLDING]，再按 period 过滤（避免相位断裂）
    global_grid = get_rebalance_days(all_dates, HOLDING)
    train_start, train_end = pd.Timestamp(TRAIN_PERIOD[0]), pd.Timestamp(TRAIN_PERIOD[1])
    valid_start, valid_end = pd.Timestamp(VALID_PERIOD[0]), pd.Timestamp(VALID_PERIOD[1])
    train_rebalance = [d for d in global_grid if train_start <= d <= train_end]
    valid_rebalance = [d for d in global_grid if valid_start <= d <= valid_end]
    print(f"📊 全局调仓网格: {len(global_grid)} 天")
    print(f"📊 训练集调仓日: {len(train_rebalance)} 天")
    print(f"📊 验证集调仓日: {len(valid_rebalance)} 天")

    # 4. IC 符号（训练集评估结果，用于对齐多空方向）
    ic_signs = load_ic_signs(suffix)

    # 5. 运行分层回测（训练集）
    print("\n" + "=" * 60)
    print("开始训练集分层回测")
    print("=" * 60)
    train_rows = []
    for i, factor in enumerate(factor_cols, 1):
        sign = ic_signs.get(factor, 1.0)
        res = run_quantile_for_factor(cs_dict, train_rebalance, factor, ic_sign=sign)
        if not res:
            print(f"⚠️ [{i}/{len(factor_cols)}] {factor} 无有效调仓日，跳过")
            continue
        row = {"factor": factor}
        for g in range(N_GROUPS):
            row[f"G{g+1}_mean"] = res["group_avg"][g]
        row.update({k: v for k, v in res.items() if k != "group_avg"})
        row.pop("cum_net", None)
        train_rows.append(row)
        if (i % 10 == 0) or (i == len(factor_cols)):
            print(f"   ⏳ [{i}/{len(factor_cols)}] 已完成")
    df_train = pd.DataFrame(train_rows)
    if not df_train.empty:
        df_train = df_train.sort_values("ann_return", ascending=False).reset_index(drop=True)

    # 6. 运行分层回测（验证集，仅观察）
    print("\n" + "=" * 60)
    print("开始验证集分层回测（仅观察）")
    print("=" * 60)
    valid_rows = []
    for i, factor in enumerate(factor_cols, 1):
        sign = ic_signs.get(factor, 1.0)
        res = run_quantile_for_factor(cs_dict, valid_rebalance, factor, ic_sign=sign)
        if not res:
            continue
        row = {"factor": factor}
        for g in range(N_GROUPS):
            row[f"G{g+1}_mean"] = res["group_avg"][g]
        row.update({k: v for k, v in res.items() if k != "group_avg"})
        row.pop("cum_net", None)
        valid_rows.append(row)
    df_valid = pd.DataFrame(valid_rows)
    if not df_valid.empty:
        df_valid = df_valid.sort_values("ann_return", ascending=False).reset_index(drop=True)

    # 7. 落盘
    train_path = DATA_PROCESSED / f"factor_quantile_train_{suffix}.parquet"
    valid_path = DATA_PROCESSED / f"factor_quantile_valid_{suffix}.parquet"
    df_train.to_parquet(train_path, index=False)
    df_valid.to_parquet(valid_path, index=False)
    print(f"✅ 训练集结果: {train_path}（{len(df_train)} 个因子）")
    print(f"✅ 验证集结果: {valid_path}（{len(df_valid)} 个因子）")

    # 8. 生成报告
    report_path = REPORTS_DIR / f"factor_quantile_report_{suffix}.txt"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["factor", "ann_return", "sharpe", "max_drawdown", "turnover", "monotonic_corr", "ic_sign"]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"因子分层回测报告（{N_GROUPS}分组，{HOLDING}日调仓，{LABEL_COL}，IC符号对齐）\n")
        f.write("=" * 80 + "\n\n")
        f.write("训练集统计（按多空收益降序，ann_return 已按 IC 符号对齐）:\n")
        if not df_train.empty:
            f.write(df_train[cols].to_string(index=False, float_format="%.4f"))
        f.write("\n\n验证集统计（按多空收益降序，ann_return 已按 IC 符号对齐）:\n")
        if not df_valid.empty:
            f.write(df_valid[cols].to_string(index=False, float_format="%.4f"))
    print(f"✅ 报告: {report_path}")

    # 9. 筛选建议（基于训练集，绝对值口径，负向因子不误杀）
    if not df_train.empty:
        passed = df_train[
            (df_train["ann_return"] > 0)
            & (df_train["monotonic_corr"].abs() > 0.8)
        ]
        print(f"\n📌 通过分层回测（多空>0 且 |单调性|>0.8）: {len(passed)} / {len(df_train)}")
        if len(passed) > 0:
            print("通过因子:", passed["factor"].tolist())

        # 落盘通过因子清单（供 train.py 特征子集使用）
        passed_path = DATA_PROCESSED / f"passed_factor_cols_{suffix}.json"
        with open(passed_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "criteria": {
                        "holding": HOLDING,
                        "n_groups": N_GROUPS,
                        "label": LABEL_COL,
                        "ann_return_min": 0.0,
                        "monotonic_corr_abs_min": 0.8,
                        "ic_sign_aligned": True,
                    },
                    "n_total": int(len(df_train)),
                    "n_passed": int(len(passed)),
                    "factors": passed["factor"].tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"✅ 通过因子清单: {passed_path}（{len(passed)} 个）")

    print("\n✅ 分层回测完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
