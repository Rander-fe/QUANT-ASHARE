"""PyTorch MLP 的 Purged Walk-Forward 训练。"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import MODELS_MLP_DIR
from models.lightgbm.config import LABEL_COL, ROLLING
from models.rolling import (feature_matrix, finish_run, load_inputs, rolling_slices,
                            sample_rows, segment_of, valid_rows)


class FactorMLP(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def predict_batches(model, values: np.ndarray, device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start:start + batch_size]).to(device)
            outputs.append(model(x).cpu().numpy())
    return np.concatenate(outputs)


def fit_model(train, valid, features, label, args, device):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(feature_matrix(train, features)),
                      torch.from_numpy(train[label].to_numpy(np.float32))),
        batch_size=args.batch_size, shuffle=True,
    )
    x_valid = feature_matrix(valid, features)
    y_valid = valid[label].to_numpy(np.float32)
    model = FactorMLP(len(features), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.HuberLoss(delta=0.05)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        pred = predict_batches(model, x_valid, device, args.batch_size)
        val_loss = float(np.mean((pred - y_valid) ** 2))
        if val_loss < best_loss - 1e-8:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss, epoch + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch MLP 滚动选股模型")
    parser.add_argument("--label", default=LABEL_COL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=400_000)
    parser.add_argument("--max-valid-samples", type=int, default=150_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-segments", default="valid")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args()
    segments = [x.strip() for x in args.eval_segments.split(",") if x.strip()]
    if "test" in segments and not args.confirm_final_test:
        parser.error("测试集评估需显式传入 --confirm-final-test")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch={torch.__version__}, device={device}")
    df, features = load_inputs(args.label)
    partial = args.quick or args.max_steps is not None
    predictions = []
    for i, train, valid, test, _, _, _ in rolling_slices(
        df, args.label, ROLLING["step"], ROLLING["train_len"], ROLLING["valid_len"],
        args.quick, args.max_steps,
    ):
        train = sample_rows(valid_rows(train, args.label), args.max_train_samples, 42 + i)
        valid = sample_rows(valid_rows(valid, args.label), args.max_valid_samples, 142 + i)
        model, val_loss, epochs = fit_model(train, valid, features, args.label, args, device)
        pred = predict_batches(model, feature_matrix(test, features), device, args.batch_size)
        frame = test[["symbol", "date", args.label]].copy()
        frame["pred"] = pred
        frame["segment"] = frame["date"].map(segment_of)
        predictions.append(frame)
        model_dir = MODELS_MLP_DIR / ("smoke" if partial else args.label) / f"step_{i:03d}"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model.pt")
        (model_dir / "metadata.json").write_text(json.dumps(
            {"features": features, "val_mse": val_loss, "epochs": epochs}, ensure_ascii=False
        ), encoding="utf-8")
        print(f"[INFO] MLP {i + 1}: val_mse={val_loss:.8f}, epochs={epochs}")
    finish_run(predictions, "mlp", args.label, segments, partial=partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
