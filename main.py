# -*- coding: utf-8 -*-
"""
流水线主程序：统一编排 A股量化 ML 选股项目的各环节。

支持两种方式：
    1. 单环节单独跑：python main.py <环节名> [额外参数...]
    2. 一键跑全流程：python main.py all

环节一览（按数据流顺序）：
    fetch_daily           从 Qlib 导出日线行情 -> data/raw/daily/
    fetch_daily_basic     按交易日拉取 daily_basic（含 pct_chg）-> basic_cleaned_with_extra_by_date.parquet
    fetch_financial       拉取财务三大报表 + fina_indicator（断点续传）-> data/raw/financial/，并自动生成事件表
    build_financial_events 仅重建 Point-in-Time 事件表（不拉数据，修复用）-> data/processed/financial/financial_events.parquet
    merge_data            合并日线 + 财务事件表 -> data/processed/merged_data.parquet
    fetch_stock_basic      拉取证券静态信息（上市日期+历史名称）-> stock_basic.parquet
    clean_data            清洗底表（指数/ST/次新/重复/停牌NaN/财务FFill）-> data/processed/basic_cleaned.parquet
    build_factors         计算全部注册因子 -> data/processed/factors.parquet
    build_alpha158        用 Qlib Alpha158 生成 158 个因子 -> alpha158.parquet
    evaluate_factors      计算 270 个因子的日频 RankIC + 汇总统计 -> factor_evaluation.parquet / daily_rankic.parquet
    remove_redundant      基于日频 IC 相关性剔除冗余因子（>0.7）-> selected_factor_cols.json
    factor_quantile       5分组分层回测（IC符号对齐，训练集把关/验证集观察）-> factor_quantile_{train,valid}.parquet
    train_lgb             LightGBM 滚动重训选股（自研因子+Alpha158，RR 基准）
    optuna_search         Optuna 超参搜索（验证集窗口，-> models/lightgbm/best_params.json）
    lgb_baseline          Qlib Alpha158 + LightGBM 基线回测（MLflow 记录）

注意：
    - fetch_daily_basic 依赖 basic_cleaned.parquet（即 clean_data 需先跑）；
      全流程中会按此顺序执行。
    - lgb_baseline 使用 Qlib 自有数据源（~/.qlib/qlib_data/cn_data），
      不依赖本项目的 parquet 产物，可独立运行。
    - train_lgb 依赖 build_factors + build_alpha158 的产物（factors.parquet /
      alpha158.parquet），在验证集（2023.5~2025.1）上评估用于模型选择，
      测试集（2025.1 之后）仅在最终评估时使用一次。

用法示例：
    python main.py build_factors
    python main.py all
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# 环节定义：名称 -> (脚本相对路径, 是否需要前置产物说明)
STAGES: dict[str, tuple[str, str]] = {
    "audit_raw_data": (
        "-m data_catalog.audit",
        "审计第0层原始数据契约 -> data/catalog/raw_data_inventory.json",
    ),
    "audit_operators": (
        "-m data_catalog.operator_audit",
        "审计第1层安全算子白名单、实现与延后项",
    ),
    "audit_quantmind_integration": (
        "-m quantmind_integration.policy",
        "审计QUANTMIND/Qlib算子复用映射、字段白名单和时间边界",
    ),
    "build_factor_catalog": (
        "-m factor_catalog.build",
        "建立第2层原子因子目录（自研 + Alpha158，不重算因子值）",
    ),
    "audit_factor_catalog": (
        "-m factor_catalog.audit",
        "审计第2层因子数量、来源、分类、输入字段与时间契约",
    ),
    "build_final_sota": (
        "-m factor_catalog.sota",
        "生成并审计人工优选58因子最终正式SOTA清单",
    ),
    "qm_mine_one_factor": (
        "-m scripts.mine_one_factor",
        "QM:DeepSeek生成1候选->近2年quick粗筛->5d全量评估+core25/准入注册表查重->10bps周频回测;通过者自动注册(仅训练期)",
    ),
    "qm_build_mining_memory": (
        "-m scripts.build_factor_mining_memory",
        "QM:重建因子挖掘经验记忆与方向地图(reports/factor_mining_memory.json)",
    ),
    "validate_reversal_fixes": (
        "analysis/validate_reversal_momentum_fixes.py --rebuild --label label_ret_20",
        "独立重算并验证4个反转/动量V2因子（不覆盖正式因子库）",
    ),
    "collect_text": (
        "-m textdata.collector",
        "采集新闻/政策/公告原始文本，按首次发现时间审计并哈希去重",
    ),
    "fetch_daily": (
        "scripts/fetch_daily.py",
        "从 Qlib 导出日线行情（all instruments）到 data/raw/daily/",
    ),
    "fetch_daily_basic": (
        "scripts/fetch_daily_basic_by_date.py",
        "按交易日拉取 daily_basic 并合并到 basic_cleaned_with_extra_by_date.parquet",
    ),
    "fetch_financial": (
        "scripts/fetch_financial.py",
        "拉取财务数据（断点续传，可重复运行）并自动生成 Point-in-Time 事件表",
    ),
    "build_financial_events": (
        "scripts/fetch_financial.py --events-only",
        "仅重建 Point-in-Time 事件表（不拉数据，修复事件表过期/缺失用）",
    ),
    "merge_data": (
        "scripts/merge_data.py",
        "合并日线 + 财务事件表 -> merged_data.parquet",
    ),
    "fetch_stock_basic": (
        "scripts/fetch_stock_basic.py",
        "拉取证券静态信息（上市日期 + 历史名称变更）-> stock_basic.parquet",
    ),
    "clean_data": (
        "scripts/clean_data.py",
        "清洗底表 -> basic_cleaned.parquet",
    ),
    "build_factors": (
        "scripts/build_factors.py",
        "计算全部注册因子 -> factors.parquet",
    ),
    "build_alpha158": (
        "scripts/build_alpha158.py",
        "用 Qlib Alpha158 生成 158 个因子 -> alpha158.parquet",
    ),
    "evaluate_factors": (
        "analysis/evaluate_factors.py",
        "计算全部因子日频 RankIC + 汇总 -> factor_evaluation.parquet / daily_rankic.parquet",
    ),
    "remove_redundant": (
        "analysis/remove_redundant_factors.py",
        "按日频 IC 相关性剔除冗余因子 -> selected_factor_cols.json / factor_removed.parquet",
    ),
    "factor_quantile": (
        "analysis/factor_quantile.py",
        "5分组分层回测（训练集把关/验证集观察，IC符号对齐）-> factor_quantile_train.parquet / factor_quantile_valid.parquet",
    ),
    "preprocess_factors": (
        "preprocessing/preprocess_factors.py",
        "因子预处理（去极值+行业/市值中性化+标准化+官方口径涨跌停标记）-> features.parquet / labels.parquet",
    ),
    "neutralize": (
        "preprocessing/neutralize.py",
        "因子中性化（行业/市值/联合，OLS 回归取残差）-> neutralized_{mode}.parquet / _summary.parquet",
    ),
    "train_lgb": (
        "models/lightgbm/train.py",
        "LightGBM 滚动重训选股（自研因子+Alpha158）-> predictions/ + models/lightgbm/",
    ),
    "optuna_search": (
        "models/lightgbm/optuna_search.py",
        "Optuna 超参搜索（验证集窗口，--n-trials 默认30，-> models/lightgbm/best_params.json）",
    ),
    "lgb_baseline": (
        "scripts/lgb_baseline.py",
        "Qlib Alpha158 + LightGBM 基线回测（MLflow 记录到 mlruns/）",
    ),
    "portfolio_backtest": (
        "backtest/run_backtest.py",
        "复合 Alpha 组合回测（默认仅验证集；测试集需显式一次性确认）",
    ),
    "tune_portfolio": (
        "analysis/tune_portfolio.py",
        "验证集内部时间留出法选择风险、收益与换手组合参数（不读取测试集）",
    ),
    "evaluate_factors_v2": (
        "analysis/evaluate_factors.py --label label_ret_20 --output-prefix v2_label20",
        "V2按20日标签重算训练期270因子日频RankIC（不覆盖V1）",
    ),
    "select_factors_v2": (
        "analysis/select_stable_factors_v2.py",
        "V2按早/中/近期稳定性与衰减约束选择因子（仅训练集）",
    ),
    "replicate_gp_huatai": (
        "analysis/replicate_gp_huatai.py",
        "复现并审计华泰遗传规划报告的6个量价因子（仅训练/验证集）",
    ),
    "optuna_search_v2": (
        "models/lightgbm/optuna_search_v2.py",
        "V2多折Purged Walk-Forward稳健LightGBM超参搜索（不覆盖V1）",
    ),
    "train_ridge": (
        "models/ridge/train.py",
        "Ridge 线性回归 Purged Walk-Forward 基线",
    ),
    "train_simple_baselines": (
        "models/baselines/train.py",
        "Purged Walk-Forward 等权与训练期IC加权因子合成基准",
    ),
    "train_mlp": (
        "models/mlp/train.py",
        "PyTorch MLP Purged Walk-Forward 模型",
    ),
    "compare_models": (
        "analysis/compare_models.py",
        "在同一验证集比较等权 / IC加权 / Ridge / LightGBM / MLP",
    ),
    "model_comparison_all": (
        "scripts/run_model_comparison.py",
        "串行训练 Ridge/MLP 并与 LightGBM 做验证集比较",
    ),
}

# 一键全流程的执行顺序
# 注：fetch_financial 已内置事件表生成，无需再单独跑 build_financial_events
ALL_ORDER = [
    "fetch_daily",
    "fetch_financial",
    "merge_data",
    "fetch_stock_basic",
    "clean_data",
    "fetch_daily_basic",
    "build_factors",
    "build_alpha158",
    "preprocess_factors",
    "neutralize",
    "factor_quantile",
    "optuna_search",
    "train_lgb",
    "lgb_baseline",
]


def run_stage(name: str, extra_args: list[str]) -> int:
    """运行单个环节，返回退出码"""
    if name not in STAGES:
        print(f"[ERROR] 未知环节: {name}")
        print(f"        可用环节: {', '.join(STAGES.keys())}")
        return 1

    script, desc = STAGES[name]
    # 支持脚本字符串带参数，如 "scripts/fetch_financial.py --events-only"
    script_parts = shlex.split(script)
    module_mode = script_parts[:1] == ["-m"]
    script_path = ROOT / script_parts[0]
    if not module_mode and not script_path.exists():
        print(f"[ERROR] 脚本不存在: {script_path}")
        return 1

    print("=" * 70)
    print(f"▶ 环节 [{name}]：{desc}")
    print("=" * 70)

    cmd = (
        [PYTHON, *script_parts, *extra_args]
        if module_mode
        else [PYTHON, str(script_path), *script_parts[1:], *extra_args]
    )
    print(f"$ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n❌ 环节 [{name}] 失败（退出码 {result.returncode}）")
    else:
        print(f"\n✅ 环节 [{name}] 完成")
    return result.returncode


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 0

    stage = args[0]
    extra = args[1:]

    if stage in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    if stage == "all":
        print("🚀 一键跑全流程（按依赖顺序执行）\n")
        failed = []
        for name in ALL_ORDER:
            code = run_stage(name, [])
            if code != 0:
                failed.append(name)
        if failed:
            print(f"\n❌ 全流程结束，失败环节: {failed}")
            return 1
        print("\n🎉 全流程全部完成！")
        return 0

    if stage == "list":
        print("可用环节：")
        for name, (_, desc) in STAGES.items():
            print(f"  {name:<24} {desc}")
        return 0

    return run_stage(stage, extra)


if __name__ == "__main__":
    sys.exit(main())
