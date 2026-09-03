# -*- coding: utf-8 -*-
"""config 包：集中管理路径、时间划分、数据源与字段定义。"""
from .settings import (
    ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    REPORTS_DIR,
    TRAIN_PERIOD,
    VALID_PERIOD,
    TEST_PERIOD,
    DATA_START,
    DATA_END,
    TUSHARE_TOKEN,
    get_tushare_token,
    gen_fin_periods,
)

__all__ = [
    "ROOT",
    "DATA_RAW",
    "DATA_PROCESSED",
    "REPORTS_DIR",
    "TRAIN_PERIOD",
    "VALID_PERIOD",
    "TEST_PERIOD",
    "DATA_START",
    "DATA_END",
    "TUSHARE_TOKEN",
    "get_tushare_token",
    "gen_fin_periods",
]
