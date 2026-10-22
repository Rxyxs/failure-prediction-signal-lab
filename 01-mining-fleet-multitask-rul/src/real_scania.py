"""The multi-task idea on real trucks: SCANIA Component X.

The synthetic CAEX fleet asks one network to predict remaining useful life and the
failure class at once, sharing a trunk. This script asks whether that pays off on
real data: SCANIA Component X (Scientific Data, 2025; researchdata.se 2024-34,
CC BY 4.0), 23,550 real heavy trucks, 2,272 of which had component X repaired during
the study, with 105 anonymised operational counters read at irregular times.

Two approaches, same features, same vehicle split:

- **separate LightGBM models**: a 5-class classifier (the official remaining-time
  windows: >48, 24-48, 12-24, 6-12, 0-6) and a regressor for remaining time;
- **one multi-task network** with the architecture of `src/models/multi_task_net.py`
  (two hidden layers of 64 with BatchNorm and dropout 0.2, one head per task) and the
  module's training protocol: at least 100 epochs, early stopping on validation loss
  (patience 15, max 300), best epoch restored.

Remaining time is only known for trucks that were repaired; for censored trucks the
regression loss is masked, as in survival data. The class is scored on the official
validation set (5,046 trucks, one readout each) with the official cost matrix and the
minimum-expected-cost decision; remaining time on held-out repaired trucks.

    python -m src.real_scania --dir data/raw/scania
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "real_scania_metrics.json"
CLASS_UPPER_BOUNDS = [(6.0, 4), (12.0, 3), (24.0, 2), (48.0, 1)]
COST_MATRIX = np.array([
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
])
CENSORED_SAMPLE = 0.10  # share of readouts kept from trucks never repaired (re-weighted)
MIN_EPOCHS, MAX_EPOCHS, PATIENCE = 100, 300, 15
SEED = 42


# ----------------------------------------------------------------------------- data

def load(directory: Path) -> dict[str, pl.DataFrame]:
    read = lambda name: pl.read_csv(directory / name, null_values=["na", "NA", ""])  # noqa: E731
    return {
        "train": read("train_operational_readouts.csv"),
        "tte": read("train_tte.csv").rename({"length_of_study_time_step": "duration", "in_study_repair": "event"}),
        "train_specs": read("train_specifications.csv"),
        "val": read("validation_operational_readouts.csv"),
        "val_labels": read("validation_labels.csv"),
        "val_specs": read("validation_specifications.csv"),
    }


def features(readouts: pl.DataFrame, specs: pl.DataFrame) -> pl.DataFrame:
    """Per readout, using only that readout and the ones before it: every counter
    divided by the truck's age (the counters are cumulative, so the raw value mostly
    measures age), the age itself, and how many readouts the truck has had so far."""
    counters = [c for c in readouts.columns if c not in ("vehicle_id", "time_step")]
    df = readouts.sort(["vehicle_id", "time_step"]).with_columns(
        pl.int_range(pl.len()).over("vehicle_id").alias("n_readouts"),
        *[(pl.col(c) / pl.col("time_step").clip(lower_bound=1.0)).alias(f"rate_{c}") for c in counters],
    ).drop(counters)
    return df.join(specs, on="vehicle_id", how="left")


def label(train: pl.DataFrame, tte: pl.DataFrame) -> pl.DataFrame:
    df = train.join(tte, on="vehicle_id")
    ttf = pl.when(pl.col("event") == 1).then(pl.col("duration") - pl.col("time_step")).otherwise(None)
    df = df.with_columns(ttf.alias("ttf"))
    cls = pl.lit(0)
    for upper, c in reversed(CLASS_UPPER_BOUNDS):
        cls = pl.when(pl.col("ttf") <= upper).then(pl.lit(c)).otherwise(cls)
    return df.with_columns(cls.cast(pl.Int8).alias("cls"))


def split_vehicles(tte: pl.DataFrame, seed: int = SEED) -> dict[str, set]:
    rng = np.random.default_rng(seed)
    ids = tte["vehicle_id"].to_numpy().copy()
    rng.shuffle(ids)
    n = len(ids)
    return {"fit": set(ids[: int(0.7 * n)]), "stop": set(ids[int(0.7 * n): int(0.8 * n)]), "hold": set(ids[int(0.8 * n):])}


def sample_rows(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """All readouts of repaired trucks, plus a share of the rest, re-weighted so the
    censored trucks keep their weight in the class loss."""
    rng = np.random.default_rng(seed)
    keep = (df["event"] == 1) | (rng.random(len(df)) < CENSORED_SAMPLE)
    out = df[keep].copy()
    out["w"] = np.where(out["event"] == 1, 1.0, 1.0 / CENSORED_SAMPLE)
    return out


# ----------------------------------------------------------------------------- decisions

def min_expected_cost(proba: np.ndarray) -> np.ndarray:
    """Action with the lowest expected cost: E[cost | predict p] = sum_t P(t) COST[t][p]."""
    return (proba @ COST_MATRIX).argmin(axis=1)


def total_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int(COST_MATRIX[y_true, y_pred].sum())


# ----------------------------------------------------------------------------- models

class MultiTaskNet(nn.Module):
    """Same trunk and heads as src/models/multi_task_net.py, without the CAEX embeddings."""

    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.2, n_classes: int = 5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
        )
        self.rul_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = self.trunk(x)
        return self.rul_head(h).squeeze(-1), self.cls_head(h)


def _tensors(X, rul, has_rul, cls, w):
    return [torch.tensor(a, dtype=t) for a, t in ((X, torch.float32), (rul, torch.float32),
                                                   (has_rul, torch.float32), (cls, torch.long), (w, torch.float32))]


def train_multitask(fit, stop, seed: int = SEED, batch_size: int = 1024, lr: float = 1e-3):
    torch.manual_seed(seed)
    model = MultiTaskNet(fit[0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(reduction="none")

    def loss_of(batch):
        X, rul, has, cls, w = batch
        rul_pred, logits = model(X)
        cls_loss = (ce(logits, cls) * w).sum() / w.sum()
        rul_loss = ((rul_pred - rul) ** 2 * has).sum() / has.sum().clamp(min=1.0)
        return rul_loss + cls_loss

    fit_t, stop_t = _tensors(*fit), _tensors(*stop)
    n = len(fit_t[0])
    best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad()
            loss = loss_of([t[idx] for t in fit_t])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_of(stop_t))
        history.append(val)
        if val < best - 1e-4:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch + 1 >= MIN_EPOCHS and bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": len(history), "best_epoch": best_epoch + 1}


# ----------------------------------------------------------------------------- run

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=ROOT / "data" / "raw" / "scania")
    args = parser.parse_args()
    start = time.perf_counter()

    d = load(args.dir)
    train = label(features(d["train"], d["train_specs"]), d["tte"]).to_pandas()
    val = features(d["val"], d["val_specs"]).sort(["vehicle_id", "time_step"]).group_by("vehicle_id").last()
    val = val.join(d["val_labels"], on="vehicle_id").to_pandas()
    spec_cols = [c for c in d["train_specs"].columns if c != "vehicle_id"]
    feat_cols = [c for c in train.columns if c.startswith("rate_")] + ["time_step", "n_readouts"]

    # Specifications: integer codes for LightGBM, one-hot for the network.
    for c in spec_cols:
        cats = pd.Categorical(pd.concat([train[c], val[c]])).categories
        train[c] = pd.Categorical(train[c], categories=cats)
        val[c] = pd.Categorical(val[c], categories=cats)

    parts = split_vehicles(d["tte"])
    fit = sample_rows(train[train.vehicle_id.isin(parts["fit"])])
    stop = sample_rows(train[train.vehicle_id.isin(parts["stop"])])
    hold = train[train.vehicle_id.isin(parts["hold"]) & (train.event == 1)]
    print(f"rows: fit {len(fit):,}, stop {len(stop):,}, held-out repaired-truck readouts {len(hold):,}, validation trucks {len(val):,}")

    y_val = val["class_label"].to_numpy()
    results: dict = {"validation_trucks": int(len(val)), "validation_failing": int((y_val > 0).sum())}
    results["recommend_nothing_cost"] = total_cost(y_val, np.zeros_like(y_val))

    # ---- separate LightGBM models
    X_cols = feat_cols + spec_cols
    clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, random_state=SEED, verbosity=-1)
    clf.fit(fit[X_cols], fit["cls"], sample_weight=fit["w"], eval_set=[(stop[X_cols], stop["cls"])],
            eval_sample_weight=[stop["w"].to_numpy()], callbacks=[lgb.early_stopping(100, verbose=False)])
    p_lgb = clf.predict_proba(val[X_cols])
    rep = fit[fit.event == 1]
    reg = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63, random_state=SEED, verbosity=-1)
    stop_rep = stop[stop.event == 1]
    reg.fit(rep[X_cols], np.log1p(rep["ttf"]), eval_set=[(stop_rep[X_cols], np.log1p(stop_rep["ttf"]))],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    rul_lgb = np.expm1(reg.predict(hold[X_cols]))

    # ---- one multi-task network
    # Medians, scaler and one-hot columns are learned on the fit split only.
    medians = fit[feat_cols].astype(float).median()
    sc = StandardScaler().fit(fit[feat_cols].astype(float).fillna(medians))
    oh_columns = pd.get_dummies(fit[spec_cols], dtype=float).columns

    def to_X(df):
        num = df[feat_cols].astype(float).fillna(medians)
        oh = pd.get_dummies(df[spec_cols], dtype=float).reindex(columns=oh_columns, fill_value=0.0)
        return np.hstack([sc.transform(num), oh.to_numpy()]).astype(np.float32)

    def targets(df):
        has = df["event"].to_numpy() == 1
        rul = np.where(has, np.log1p(df["ttf"].fillna(0).clip(lower=0).to_numpy()), 0.0)
        return rul, has.astype(float), df["cls"].to_numpy(), df["w"].to_numpy()

    net, net_info = train_multitask((to_X(fit), *targets(fit)), (to_X(stop), *targets(stop)))
    with torch.no_grad():
        rul_net_log, logits = net(torch.tensor(to_X(hold)))
        _, val_logits = net(torch.tensor(to_X(val)))
    rul_net = np.expm1(rul_net_log.numpy())
    p_net = torch.softmax(val_logits, dim=1).numpy()

    # ---- scores
    y_hold = hold["ttf"].to_numpy()
    near = y_hold <= 48
    baseline = np.full_like(y_hold, np.median(rep["ttf"]), dtype=float)
    for name, proba, rul in (("lightgbm_separate", p_lgb, rul_lgb), ("multitask_network", p_net, rul_net)):
        decision = min_expected_cost(proba)
        results[name] = {
            "validation_cost_min_expected": total_cost(y_val, decision),
            "validation_any_risk_auc": round(float(roc_auc_score(y_val > 0, 1 - proba[:, 0])), 4),
            "failing_trucks_alarmed": int(((decision > 0) & (y_val > 0)).sum()),
            "healthy_trucks_alarmed": int(((decision > 0) & (y_val == 0)).sum()),
            "rul_mae_all_readouts": round(float(mean_absolute_error(y_hold, rul)), 1),
            "rul_mae_last_48": round(float(mean_absolute_error(y_hold[near], rul[near])), 1),
        }
        print(name, results[name])
    results["multitask_network"].update(net_info)
    results["rul_median_baseline"] = {"rul_mae_all_readouts": round(float(mean_absolute_error(y_hold, baseline)), 1),
                                      "rul_mae_last_48": round(float(mean_absolute_error(y_hold[near], baseline[near])), 1)}
    results["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
