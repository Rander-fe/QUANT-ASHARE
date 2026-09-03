# -*- coding: utf-8 -*-
"""集中配置：路径、时间划分、数据源、财务字段。

所有时间划分遵循项目铁律（见 .github/copilot-instructions.md），
严禁在训练/验证/测试之间混用数据。

token 读取优先级：环境变量 TUSHARE_TOKEN > 根目录 .env 文件。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FACTORS_DIR = ROOT / "factors"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# 模型 / 预测 / 报告子目录（LightGBM 滚动重训产物）
MODELS_LGB_DIR = MODELS_DIR / "lightgbm"
MODELS_RIDGE_DIR = MODELS_DIR / "ridge"
MODELS_MLP_DIR = MODELS_DIR / "mlp"
LGB_REPORTS_DIR = REPORTS_DIR / "lightgbm"
PREDICTIONS_DIR = DATA_PROCESSED / "predictions"

# 组合构建 / 回测产物。验证集研究与最终测试严格分目录保存。
BACKTEST_DIR = ROOT / "backtest"
BACKTEST_REPORTS_DIR = REPORTS_DIR / "portfolio"
TEST_EVALUATION_LOCK = BACKTEST_REPORTS_DIR / "final_test_audit.json"
SELECTED_MODEL_PATH = DATA_PROCESSED / "selected_model.json"

# 因子预处理产物（preprocess_factors 环节）
FEATURES_FILE = "features.parquet"
LABELS_FILE = "labels.parquet"
# 含行业/市值/涨跌停标记的底表（fetch_daily_basic_by_date.py 产物）
BASIC_EXTRA_PATH = DATA_PROCESSED / "basic_cleaned_with_extra_by_date.parquet"

# 财务数据原始落盘目录
FIN_RAW_DIR = DATA_RAW / "financial"
FIN_PROCESSED_DIR = DATA_PROCESSED / "financial"

# 证券静态信息（fetch_stock_basic.py 产物）：上市日期 + 历史名称变更
STOCK_BASIC_PATH = DATA_PROCESSED / "stock_basic.parquet"
NAMECHANGE_PATH = DATA_PROCESSED / "namechange.parquet"

# Qlib 二进制行情库路径（含 6135 只 A 股 2000-2026 日线，字段见 DAILY_FIELDS）
QLIB_PROVIDER_URI = os.environ.get(
    "QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"
)
# 从 Qlib 导出的日线行情落盘目录（长表 parquet，按交易所前缀 sh/sz/bj 分文件）
DAILY_RAW_DIR = DATA_RAW / "daily"

# ---------------------------------------------------------------------------
# 时间划分（铁律，不可违反）
# ---------------------------------------------------------------------------
# 训练集：其余历史数据，采用滚动训练（rolling retraining）
TRAIN_PERIOD = ("2016-01-01", "2023-04-30")
# 验证集：仅用于超参/模型/因子选择
VALID_PERIOD = ("2023-05-01", "2025-01-01")
# 测试集：严禁用于反向选择方案，只在最终评估时使用一次
TEST_PERIOD = ("2025-01-02", "2026-08-13")

# 数据整体区间
DATA_START = "2016-01-01"
DATA_END = "2026-12-31"

# ---------------------------------------------------------------------------
# 财务数据：报告期
# ---------------------------------------------------------------------------
FIN_START_YEAR = 2016
FIN_END_YEAR = 2026
# 季报/中报/年报的报告期末日期（月日）
FIN_PERIOD_MD = ("0331", "0630", "0930", "1231")
# 仅保留合并报表（report_type=1），避免母公司/单季等重复口径
FIN_REPORT_TYPE = "1"


def gen_fin_periods(start_year: int = FIN_START_YEAR, end_year: int = FIN_END_YEAR):
    """生成报告期列表，如 ['20160331', ..., '20261231']。

    报告期为每个季度最后一天的日期：0331 一季报 / 0630 中报 / 0930 三季报 / 1231 年报。
    """
    periods = []
    for year in range(start_year, end_year + 1):
        for md in FIN_PERIOD_MD:
            periods.append(f"{year}{md}")
    return periods


# ---------------------------------------------------------------------------
# tushare token
# ---------------------------------------------------------------------------
def get_tushare_token() -> str:
    """从环境变量或根目录 .env 读取 TUSHARE_TOKEN，返回去除空白后的字符串。"""
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "TUSHARE_TOKEN":
                return value.strip().strip('"').strip("'")
    return ""


TUSHARE_TOKEN = get_tushare_token()


# ---------------------------------------------------------------------------
# 财务三大报表：tushare 接口名 -> 落盘文件名
# ---------------------------------------------------------------------------
FINANCIAL_STATEMENTS = {
    # 利润表（income statement）
    "income": "income.parquet",
    # 资产负债表（balance sheet）
    "balancesheet": "balancesheet.parquet",
    # 现金流量表（cash flow statement）
    "cashflow": "cashflow.parquet",
}
