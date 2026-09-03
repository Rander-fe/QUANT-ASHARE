# -*- coding: utf-8 -*-
"""数据加载与滚动窗口划分。

职责：
    1. 加载 features.parquet（预处理后：去极值+行业/市值中性化+标准化）
       与 labels.parquet（原值标签）
    2. 剔除指数/B 股，按 (symbol, date) 合并为特征宽表
    3. 按 Rolling Retrain（RR）生成滚动窗口：
       每步 [train, valid, test]，test 段永远在 train/valid 之后（无未来函数）

所有路径均从 config.settings 读取，禁止硬编码。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import DATA_PROCESSED, MODELS_LGB_DIR
from models.lightgbm.config import (
    ALT_LABEL_COLS,
    BEST_PARAMS_FILE,
    FACTORS_FILE,
    FACTOR_LIST_FILE,
    KEY_COLS,
    LABEL_COL,
    LABELS_FILE,
    NON_FEATURE_COLS,
    NON_STOCK_PREFIXES,
    ROLLING,
)


def _parquet_columns(path: Path) -> list[str]:
    """读取 parquet 的实际列名（不加载数据）。"""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema.names


def load_factor_list(filename: str | None = None,
                     label_col: str | None = None,
                     strict: bool = True) -> list[str] | None:
    """加载精选因子清单（分层回测通过因子）。

    Returns
    -------
    list[str] | None
        因子清单。默认缺失时直接报错，防止不同标签在不知情的情况下
        使用不同特征全集；仅探索性任务可显式传入 ``strict=False``。
    """
    if filename:
        path = DATA_PROCESSED / filename
    elif label_col:
        path = DATA_PROCESSED / f"passed_factor_cols_ret{label_horizon(label_col)}.json"
    else:
        path = DATA_PROCESSED / FACTOR_LIST_FILE
    if not path.exists():
        # 兼容旧产物，但只允许元数据明确匹配当前标签，绝不静默跨标签复用。
        legacy = DATA_PROCESSED / FACTOR_LIST_FILE
        if label_col and legacy.exists():
            with open(legacy, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            if legacy_data.get("criteria", {}).get("label") == label_col:
                print(f"[WARN] 使用匹配标签的旧版因子清单: {legacy}")
                return list(legacy_data["factors"])
        if strict:
            raise FileNotFoundError(
                f"精选因子清单不存在: {path}。受控实验禁止静默使用全部特征；"
                "请先生成匹配标签的清单，或显式调用 strict=False。"
            )
        print(f"[WARN] 精选因子清单不存在: {path}，探索模式使用全部特征列")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    actual_label = data.get("criteria", {}).get("label")
    if label_col and actual_label != label_col:
        raise ValueError(
            f"因子清单标签不匹配: 请求 {label_col}，但 {path.name} 记录为 {actual_label}"
        )
    return list(data["factors"])


def load_best_params(label_col: str = LABEL_COL) -> dict | None:
    """加载 Optuna 搜索得到的最优超参（按标签取）。

    Parameters
    ----------
    label_col : 标签列名（best_params.json 按标签分条存储）

    Returns
    -------
    dict | None
        最优超参（覆盖 DEFAULT_LGB_PARAMS）；无该标签的记录返回 None。
    """
    path = MODELS_LGB_DIR / BEST_PARAMS_FILE
    if not path.exists():
        print(f"[WARN] 最优超参文件不存在: {path}，使用默认超参")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容旧单条格式 {"label": ..., "params": ...}
    if "label" in data and "params" in data:
        data = {data["label"]: data}
    record = data.get(label_col)
    if not isinstance(record, dict):
        print(f"[WARN] {path} 无标签 {label_col} 的记录（现有: {list(data.keys())}），"
              f"使用默认超参")
        return None
    params = record.get("params")
    if not isinstance(params, dict):
        print(f"[WARN] {path} 的 {label_col} 记录缺少 params 字段，使用默认超参")
        return None
    print(f"[INFO] 使用 Optuna 最优超参（{label_col}，"
          f"valid RankIC={record.get('best_rank_ic', '?')}）")
    return params


def load_feature_tables(columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载预处理产物。

    Parameters
    ----------
    columns : 仅读取这些列（大幅降低内存），None = 读全部列

    Returns
    -------
    (features_df, labels_df)
        features_df : 中性化+标准化特征宽表 [symbol, date, 特征..., 标记...]
        labels_df   : 原值标签 [symbol, date, label_ret_5/10/20]
    """
    features_path = DATA_PROCESSED / FACTORS_FILE
    labels_path = DATA_PROCESSED / LABELS_FILE
    if not features_path.exists():
        raise FileNotFoundError(
            f"缺少预处理特征表 {features_path}，请先运行 main.py preprocess_factors"
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"缺少标签表 {labels_path}，请先运行 main.py preprocess_factors"
        )

    # 只读取需要的列（精选因子场景下 270 列全读会 OOM）
    if columns is not None:
        features_df = pd.read_parquet(features_path, columns=columns)
    else:
        features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    return features_df, labels_df


def drop_non_stock(df: pd.DataFrame) -> pd.DataFrame:
    """剔除指数 / B 股等非 A 股标的。"""
    mask = ~df["symbol"].str.startswith(NON_STOCK_PREFIXES)
    return df[mask].copy()


def merge_features(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """合并特征与标签为训练宽表，返回 [symbol, date, 特征..., 标记..., label]。"""
    # 剔除非股票
    features_df = drop_non_stock(features_df)
    labels_df = drop_non_stock(labels_df)

    # 统一日期类型
    features_df["date"] = pd.to_datetime(features_df["date"])
    labels_df["date"] = pd.to_datetime(labels_df["date"])

    # 特征列：除主键与标记列外全部
    feat_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
    # 标签列：labels 表中实际存在的（默认 label_ret_20，缺失则回退）
    label_cols = [c for c in ALT_LABEL_COLS if c in labels_df.columns]
    if LABEL_COL not in label_cols:
        raise ValueError(f"labels.parquet 缺少标签列 {LABEL_COL}，现有标签: {label_cols}")

    # 保留的标记列（涨跌停等）
    meta_cols = [c for c in features_df.columns if c in NON_FEATURE_COLS and c not in KEY_COLS]

    # inner 合并：保证两表股票/日期对齐
    merged = features_df[KEY_COLS + meta_cols + feat_cols].merge(
        labels_df[KEY_COLS + label_cols],
        on=KEY_COLS,
        how="inner",
    )
    print(f"[INFO] 合并后: {len(merged):,} 行, {merged['symbol'].nunique()} 只, "
          f"{merged['date'].nunique()} 个交易日, {len(feat_cols)} 个特征")
    return merged


def get_feature_cols(df: pd.DataFrame, label_col: str) -> list[str]:
    """特征列 = 除主键、标记列与标签外的所有列。"""
    # 合并表会同时携带 5/10/20 日标签；必须排除全部标签，而非只排除当前目标，
    # 否则其他未来收益会被当作特征，形成直接的未来信息泄漏。
    excluded = set(NON_FEATURE_COLS) | set(ALT_LABEL_COLS)
    return [c for c in df.columns if c not in excluded]


def prepare_data(label_col: str = LABEL_COL, factor_list: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """加载并预处理特征表。

    Parameters
    ----------
    label_col : 标签列名
    factor_list : 精选因子清单；None = 使用 features 表全部特征列

    Returns
    -------
    (df, feature_cols)
        df           : [symbol, date, 特征..., label]（按 symbol/date 排序）
        feature_cols : 特征列名列表
    """
    # 只读精选因子 + 主键 + 标记列（避免 270 列全读 OOM）
    read_cols: list[str] | None = None
    if factor_list is not None:
        # 标记列可能部分未落盘（如 pct_chg 已剔除），按实际 schema 过滤
        actual_cols = _parquet_columns(DATA_PROCESSED / FACTORS_FILE)
        meta_cols = [c for c in NON_FEATURE_COLS if c not in KEY_COLS and c in actual_cols]
        read_cols = list(KEY_COLS) + meta_cols + list(factor_list)
    features_df, labels_df = load_feature_tables(columns=read_cols)
    df = merge_features(features_df, labels_df)

    if label_col not in df.columns:
        raise ValueError(f"标签列 {label_col} 不在合并表中")

    feature_cols = get_feature_cols(df, label_col)
    # 若指定精选因子清单，则按清单过滤；未命中的因子告警
    if factor_list is not None:
        missing = [c for c in factor_list if c not in df.columns]
        if missing:
            raise ValueError(f"精选因子清单中 {len(missing)} 个因子不在 features 表: {missing[:10]}")
        feature_cols = [c for c in feature_cols if c in factor_list]
        print(f"[INFO] 精选因子清单过滤: {len(factor_list)} -> {len(feature_cols)} 个特征")

    # 缺失值填 0（预处理已做中性化/标准化，残余 NaN 用 0 兜底）
    # 注意：用原地 nan_to_num(copy=False) 而非 fillna——fillna 会复制整张
    # 1000万×270 的 float32 大表（~10GB），原地修改底层数组不产生副本。
    arr = df[feature_cols].values  # 视图，float32
    np.nan_to_num(arr, copy=False, nan=0.0)

    df = df.sort_values(KEY_COLS).reset_index(drop=True)
    return df, feature_cols


def label_horizon(label_col: str) -> int:
    """从 ``label_ret_N`` 提取前瞻期，未知标签拒绝静默回退。"""
    prefix = "label_ret_"
    if not label_col.startswith(prefix):
        raise ValueError(f"无法从标签名解析前瞻期: {label_col}")
    try:
        horizon = int(label_col[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"无法从标签名解析前瞻期: {label_col}") from exc
    if horizon <= 0:
        raise ValueError(f"标签前瞻期必须为正数: {label_col}")
    return horizon


def build_rolling_windows(
    dates: np.ndarray,
    rolling: dict | None = None,
    purge_days: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """按 Rolling Retrain 生成滚动窗口。

    Parameters
    ----------
    dates : 全部交易日（升序）
    rolling : 配置，含 step / train_len / valid_len

    Returns
    -------
    windows : [(train_idx, valid_idx, test_idx), ...]
        每步 test 段长度 = step（最后一段可不足），
    valid = test 前 valid_len 天，train = valid 前 train_len 天。
    train/valid 与 valid/test 之间各留 ``purge_days`` 个交易日，避免
    前瞻收益标签跨越集合边界（purged walk-forward split）。
    """
    if rolling is None:
        rolling = ROLLING
    step = int(rolling["step"])
    train_len = int(rolling["train_len"])
    valid_len = int(rolling["valid_len"])
    purge_days = int(purge_days)
    if purge_days < 0:
        raise ValueError("purge_days 不能为负数")

    n = len(dates)
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # 从 train_len + valid_len 之后开始滚动（保证首段有足够历史）
    start = train_len + valid_len + 2 * purge_days
    for test_end in range(start + step, n + 1, step):
        test_start = test_end - step
        valid_end = test_start - purge_days
        valid_start = valid_end - valid_len
        train_end = valid_start - purge_days
        train_start = train_end - train_len
        windows.append((
            np.arange(train_start, train_end),     # train
            np.arange(valid_start, valid_end),     # valid（早停）
            np.arange(test_start, test_end),       # test（预测）
        ))

    # 末尾不足 step 的残余段（若仍有数据）
    if start + step <= n - 1 and (n - 1) % step != 0:
        pass  # 残余段过短，忽略，避免无意义预测

    return windows
