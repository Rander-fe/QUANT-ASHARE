"""在同一验证集上公平比较 Ridge、LightGBM 与 PyTorch MLP。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (DATA_PROCESSED, PREDICTIONS_DIR, REPORTS_DIR,
                             SELECTED_MODEL_PATH, TEST_PERIOD, VALID_PERIOD)
from models.lightgbm.evaluate import evaluate_predictions
from research.protocol import write_experiment_manifest

MODELS = ("equal_weight", "ic_weight", "ridge", "lgb", "mlp")


def main() -> int:
    parser = argparse.ArgumentParser(description="三模型同标签对比")
    parser.add_argument("--label", default="label_ret_5")
    parser.add_argument("--segment", default="valid", choices=("valid", "test"))
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    if args.segment == "test" and not args.confirm_final_test:
        parser.error("测试集对比只允许最终评估，请显式确认")

    period = VALID_PERIOD if args.segment == "valid" else TEST_PERIOD
    labels = pd.read_parquet(
        DATA_PROCESSED / "labels.parquet", columns=["symbol", "date", args.label],
        filters=[("date", ">=", pd.Timestamp(period[0])), ("date", "<=", pd.Timestamp(period[1]))],
    )
    rows = []
    for model_name in MODELS:
        path = PREDICTIONS_DIR / f"{model_name}_pred_{args.label}.parquet"
        if not path.exists():
            print(f"[SKIP] {model_name}: 缺少 {path}")
            continue
        pred = pd.read_parquet(path, filters=[("segment", "==", args.segment)])
        if pred.empty:
            print(f"[SKIP] {model_name}: 没有 {args.segment} 段完整预测")
            continue
        frame = pred.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")
        metrics = evaluate_predictions(frame, "pred", args.label)
        rows.append({"model": model_name, "label": args.label, "segment": args.segment, **metrics})
        print(f"[{model_name}] RankIC={metrics['rank_ic_mean']} IC={metrics['ic_mean']} "
              f"RankICIR={metrics['rank_icir']} 多空={metrics['long_short_spread']}")
    if not rows:
        return 1
    completed = {row["model"] for row in rows}
    missing = set(MODELS) - completed
    if missing:
        print(f"[ERROR] 公平比较要求全部基准/模型完整覆盖，当前缺少: {sorted(missing)}")
        return 1
    result = pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False)
    out = REPORTS_DIR / "model_comparison" / f"{args.segment}_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    write_experiment_manifest(
        out.with_suffix(".manifest.json"),
        experiment="model_comparison",
        config={"label": args.label, "segment": args.segment,
                "selection_metric": "rank_ic_mean", "models": list(MODELS),
                "period": list(period)},
        inputs=[DATA_PROCESSED / "features.parquet", DATA_PROCESSED / "labels.parquet",
                *[PREDICTIONS_DIR / f"{name}_pred_{args.label}.parquet" for name in MODELS]],
        test_data_used=args.segment == "test",
    )
    if args.segment == "valid":
        import json
        best = result.iloc[0]
        SELECTED_MODEL_PATH.write_text(json.dumps({
            "model": str(best["model"]), "label": args.label,
            "selection_metric": "valid_rank_ic",
            "valid_rank_ic": float(best["rank_ic_mean"]),
            "comparison_report": str(out),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 已选模型配置: {SELECTED_MODEL_PATH}")
    print(f"[OK] 对比报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
