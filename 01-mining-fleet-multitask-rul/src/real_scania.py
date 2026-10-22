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

Both models are run on two feature sets. The dataset multiplies every variable by a
factor that differs between the training and the validation split (found in
heavy-truck-predictive-maintenance), so ``rates`` (counter / age) carry that factor and
``scale_invariant`` (histogram bin fractions and the recent-to-lifetime rate ratio)
cancel it. A LightGBM trained to tell training readouts from validation readouts
measures how far apart the two splits look with each set (adversarial AUC).

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
WINDOW = 5  # readouts used for the recent rate
EPS = 1e-9
FIGURE_PATH = ROOT / "outputs" / "reports" / "real_scania_feature_sets.png"


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


def histogram_groups(counters: list[str]) -> dict[str, list[str]]:
    """Variables with several bins (``397_0 .. 397_35``): the histograms of the dataset."""
    groups: dict[str, list[str]] = {}
    for c in counters:
        groups.setdefault(c.rsplit("_", 1)[0], []).append(c)
    return {k: v for k, v in groups.items() if len(v) > 1}


def features(readouts: pl.DataFrame, specs: pl.DataFrame) -> pl.DataFrame:
    """Per readout, using only that readout and the ones before it.

    * ``rate_<c>``: the counter divided by the truck's age (the counters are cumulative,
      so the raw value mostly measures age). Carries the per-split scale factor.
    * ``rel_<c>``: the rate over the last ``WINDOW`` readouts divided by ``rate_<c>``.
    * ``frac_<c>``: for histogram bins, the bin's share of its histogram's total.
    * ``time_step`` and ``n_readouts`` (how many readouts the truck has had so far).

    ``rel_`` and ``frac_`` do not change if a variable is multiplied by a constant.
    """
    counters = [c for c in readouts.columns if c not in ("vehicle_id", "time_step")]
    vid, t = "vehicle_id", pl.col("time_step")
    t_back = pl.coalesce([t.shift(WINDOW).over(vid), t.first().over(vid)])
    exprs = [pl.int_range(pl.len()).over(vid).alias("n_readouts")]
    for c in counters:
        x = pl.col(c)
        life = x / t.clip(lower_bound=1.0)
        x_back = pl.coalesce([x.shift(WINDOW).over(vid), x.first().over(vid)])
        recent = pl.when((t - t_back) > EPS).then((x - x_back) / (t - t_back)).otherwise(None)
        exprs += [life.alias(f"rate_{c}"), pl.when(life.abs() > EPS).then(recent / life).otherwise(None).alias(f"rel_{c}")]
    for cols in histogram_groups(counters).values():
        total = pl.sum_horizontal(cols)
        exprs += [pl.when(total > EPS).then(pl.col(c) / total).otherwise(None).alias(f"frac_{c}") for c in cols]
    df = readouts.sort(["vehicle_id", "time_step"]).with_columns(exprs).drop(counters)
    return df.join(specs, on="vehicle_id", how="left")


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    base = ["time_step", "n_readouts"]
    return {"rates": [c for c in columns if c.startswith("rate_")] + base,
            "scale_invariant": [c for c in columns if c.startswith(("rel_", "frac_"))] + base}


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame, seed: int = SEED) -> float:
    """5-fold AUC of a LightGBM that tries to tell rows of ``a`` from rows of ``b`` (0.5 = same distribution)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = pd.concat([a, b], ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, colsample_bytree=0.3, verbosity=-1, random_state=seed)
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 3)


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

def run_feature_set(train: pd.DataFrame, val: pd.DataFrame, parts: dict, feat_cols: list[str], spec_cols: list[str]) -> dict:
    """Separate LightGBM models and the multi-task network on one feature set."""
    train = train.replace([np.inf, -np.inf], np.nan)
    val = val.replace([np.inf, -np.inf], np.nan)
    fit = sample_rows(train[train.vehicle_id.isin(parts["fit"])])
    stop = sample_rows(train[train.vehicle_id.isin(parts["stop"])])
    hold = train[train.vehicle_id.isin(parts["hold"]) & (train.event == 1)]
    y_val = val["class_label"].to_numpy()

    # ---- separate LightGBM models
    X_cols = feat_cols + spec_cols
    clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, random_state=SEED, verbosity=-1)
    clf.fit(fit[X_cols], fit["cls"], sample_weight=fit["w"], eval_X=(stop[X_cols],), eval_y=(stop["cls"],),
            eval_sample_weight=[stop["w"].to_numpy()], callbacks=[lgb.early_stopping(100, verbose=False)])
    p_lgb = clf.predict_proba(val[X_cols])
    rep = fit[fit.event == 1]
    reg = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63, random_state=SEED, verbosity=-1)
    stop_rep = stop[stop.event == 1]
    reg.fit(rep[X_cols], np.log1p(rep["ttf"]), eval_X=(stop_rep[X_cols],), eval_y=(np.log1p(stop_rep["ttf"]),),
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
        return np.nan_to_num(np.hstack([sc.transform(num), oh.to_numpy()]), nan=0.0).clip(-50, 50).astype(np.float32)

    def targets(df):
        has = df["event"].to_numpy() == 1
        rul = np.where(has, np.log1p(df["ttf"].fillna(0).clip(lower=0).to_numpy()), 0.0)
        return rul, has.astype(float), df["cls"].to_numpy(), df["w"].to_numpy()

    net, net_info = train_multitask((to_X(fit), *targets(fit)), (to_X(stop), *targets(stop)))
    with torch.no_grad():
        rul_net_log, _ = net(torch.tensor(to_X(hold)))
        _, val_logits = net(torch.tensor(to_X(val)))
    rul_net = np.expm1(rul_net_log.numpy())
    p_net = torch.softmax(val_logits, dim=1).numpy()

    # ---- scores
    y_hold = hold["ttf"].to_numpy()
    near = y_hold <= 48
    out = {"n_features": len(feat_cols)}
    for name, proba, rul in (("lightgbm_separate", p_lgb, rul_lgb), ("multitask_network", p_net, rul_net)):
        decision = min_expected_cost(proba)
        out[name] = {
            "validation_cost_min_expected": total_cost(y_val, decision),
            "validation_any_risk_auc": round(float(roc_auc_score(y_val > 0, 1 - proba[:, 0])), 4),
            "mean_predicted_risk": round(float((1 - proba[:, 0]).mean()), 4),
            "failing_trucks_alarmed": int(((decision > 0) & (y_val > 0)).sum()),
            "healthy_trucks_alarmed": int(((decision > 0) & (y_val == 0)).sum()),
            "rul_mae_all_readouts": round(float(mean_absolute_error(y_hold, rul)), 1),
            "rul_mae_last_48": round(float(mean_absolute_error(y_hold[near], rul[near])), 1),
        }
        print(name, out[name], flush=True)
    out["multitask_network"].update(net_info)
    baseline = np.full_like(y_hold, np.median(rep["ttf"]), dtype=float)
    out["rul_median_baseline"] = {"rul_mae_all_readouts": round(float(mean_absolute_error(y_hold, baseline)), 1),
                                  "rul_mae_last_48": round(float(mean_absolute_error(y_hold[near], baseline[near])), 1)}
    return out


def make_figure(results: dict, path: Path = FIGURE_PATH) -> Path:
    """Official cost and any-risk AUC of both models on both feature sets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [("rates", "Rates (counter / age)"), ("scale_invariant", "Scale-invariant")]
    models = [("lightgbm_separate", "Separate LightGBM", "#2a78d6"), ("multitask_network", "Multi-task network", "#eb6834")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#fcfcfb")
    for ax, metric, title in ((axes[0], "validation_cost_min_expected", "Official validation cost (lower is better)"),
                              (axes[1], "validation_any_risk_auc", "Any-risk AUC on validation")):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e4e3df")
        ax.set_axisbelow(True)
        x = np.arange(len(sets))
        for i, (key, label, color) in enumerate(models):
            vals = [results["feature_sets"][s][key][metric] for s, _ in sets]
            bars = ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" if v > 1 else f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        if "cost" in metric:
            ax.axhline(results["recommend_nothing_cost"], color="#52514e", linewidth=1)
            ax.set_ylim(0, results["recommend_nothing_cost"] * 1.12)
            ax.text(-0.45, results["recommend_nothing_cost"], " recommend nothing", va="bottom", fontsize=8, color="#52514e")
        else:
            ax.axhline(0.5, color="#52514e", linewidth=1)
            ax.set_ylim(0.4, max(0.8, ax.get_ylim()[1]))
        ax.set_xticks(x, [label for _, label in sets])
        ax.set_title(title, loc="left", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(1.08, -0.1), ncol=2)
    fig.suptitle("SCANIA Component X: the same two models on two feature sets", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    return path


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
    sets = feature_sets(list(train.columns))

    # Specifications: integer codes for LightGBM, one-hot for the network.
    for c in spec_cols:
        cats = pd.Categorical(pd.concat([train[c], val[c]])).categories
        train[c] = pd.Categorical(train[c], categories=cats)
        val[c] = pd.Categorical(val[c], categories=cats)

    parts = split_vehicles(d["tte"])
    y_val = val["class_label"].to_numpy()
    results: dict = {"validation_trucks": int(len(val)), "validation_failing": int((y_val > 0).sum()),
                     "recommend_nothing_cost": total_cost(y_val, np.zeros_like(y_val)), "feature_sets": {}}
    # One random training readout per truck, the same protocol as the validation set, for the adversarial check.
    rng = np.random.default_rng(SEED)
    one = train.assign(_u=rng.random(len(train))).sort_values("_u").groupby("vehicle_id").head(1)
    for name, cols in sets.items():
        print(f"== {name}: {len(cols)} features", flush=True)
        out = run_feature_set(train, val, parts, cols, spec_cols)
        out["adversarial_auc_train_vs_validation"] = adversarial_auc(one[cols], val[cols])
        results["feature_sets"][name] = out
        print(name, "adversarial AUC", out["adversarial_auc_train_vs_validation"], flush=True)
    results["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    make_figure(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
