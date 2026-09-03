# -*- coding: utf-8 -*-
"""
Minimal end-to-end LightGBM baseline for A-share ML stock selection.

Pipeline: Alpha158 features -> LGBModel (mse) -> TopkDropoutStrategy backtest.

Time split (iron rules):
    train: 2016-01-01 ~ 2023-04-30  (rolling retraining in Stage 1)
    valid: 2023-05-01 ~ 2025-01-01  (early-stopping / model & factor selection)
    test : 2025-01-02 ~ 2026-08-13  (final eval, used only once)

Run:
    C:/Users/haoran/miniconda3/envs/rqalpha/python.exe scripts/lgb_baseline.py
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Silence the noisy "Gym has been unmaintained since 2022" banner that gym 0.26.2
# prints to stderr on every import (qlib's strategy chain imports it repeatedly).
try:
    import gym_notices.notices as _gn
    _gn.notices = {}
except Exception:
    pass

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R, QlibRecorder
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord


# --- Monkey patches for qlib 0.9.7 ---
# `record_temp.py` still calls a few APIs on the global `QlibRecorder` (R) that
# were removed from it but still live on the active `Recorder` instance.
# Route them to the active recorder.
def _recorder_experiment_id(self):
    return self.get_recorder().experiment_id


def _recorder_list_artifacts(self, artifact_path=None):
    return self.get_recorder().list_artifacts(artifact_path)


QlibRecorder.experiment_id = property(_recorder_experiment_id)
QlibRecorder.list_artifacts = _recorder_list_artifacts

PROVIDER_URI = "~/.qlib/qlib_data/cn_data"
MARKET = "csi300"
BENCHMARK = "SH000300"

TRAIN_PERIOD = ("2016-01-01", "2023-04-30")
VALID_PERIOD = ("2023-05-01", "2025-01-01")
TEST_PERIOD = ("2025-01-02", "2026-08-13")


def main():
    qlib.init(provider_uri=PROVIDER_URI, region="cn")

    handler_conf = {
        "start_time": TRAIN_PERIOD[0],
        "end_time": TEST_PERIOD[1],
        "fit_start_time": TRAIN_PERIOD[0],
        "fit_end_time": VALID_PERIOD[1],
        "instruments": MARKET,
    }

    dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": handler_conf,
            },
            "segments": {
                "train": TRAIN_PERIOD,
                "valid": VALID_PERIOD,
                "test": TEST_PERIOD,
            },
        },
    }

    model_conf = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "num_boost_round": 200,
            "early_stopping_rounds": 50,
            "learning_rate": 0.05,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
        },
    }

    dataset = init_instance_by_config(dataset_conf)
    model = init_instance_by_config(model_conf)

    with R.start(experiment_name="lgb_baseline"):
        R.log_params(**{"market": MARKET, "benchmark": BENCHMARK})

        # train (uses valid segment for early stopping)
        model.fit(dataset)
        R.save_objects(trained_model=model)

        # prediction on test segment
        sr = SignalRecord(model, dataset, R)
        sr.generate()

        # signal analysis (IC / RankIC / group return)
        sar = SigAnaRecord(R, ana_long_short=False, ann_scaler=252)
        sar.generate()

        # portfolio backtest
        port_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": "<PRED>",
                    "topk": 50,
                    "n_drop": 5,
                },
            },
            "backtest": {
                "start_time": TEST_PERIOD[0],
                "end_time": TEST_PERIOD[1],
                "account": 100000000,
                "benchmark": BENCHMARK,
                "exchange_kwargs": {
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }
        par = PortAnaRecord(R, config=port_config)
        par.generate()

        recorder = R.get_recorder()
        exp_id = recorder.experiment_id
        recorder_id = recorder.id

    print("\n[OK] LightGBM baseline finished.")
    print(f"Experiment id: {exp_id}")
    print(f"Recorder id: {recorder_id}")
    print("Check artifacts under mlruns/ for metrics & figures.")


if __name__ == "__main__":
    main()
