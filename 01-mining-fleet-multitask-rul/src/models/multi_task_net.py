"""Red multi-task en PyTorch (tronco compartido) para RUL + clasificacion de
tipo de falla, entrenada y evaluada como alternativa comparable a los dos
modelos LightGBM independientes de `train_survival_pipeline.py`.

Un tronco denso comparte representacion entre ambas tareas: procesa
features numericas de degradacion + embeddings de `equipment_type`/`faena`,
y dos cabezas lineales predicen RUL (regresion) y tipo de falla
(clasificacion multiclase) a partir de esa representacion compartida.

Ejecutar desde la raiz del repositorio con:
    python -m src.models.multi_task_net
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.mining_data_generator import EQUIPMENT_TYPES, FAENAS, FAILURE_TYPES
from src.features.engineering import FEATURE_COLUMNS
from src.models.train_survival_pipeline import (
    MODELS_DIR,
    PROCESSED_DIR,
    build_rul_training_table,
    equipment_train_test_split,
)

NUMERIC_COLUMNS = FEATURE_COLUMNS + ["operating_hours", "age_years"]
EQUIPMENT_TYPE_TO_IDX = {name: i for i, name in enumerate(EQUIPMENT_TYPES)}
FAENA_TO_IDX = {name: i for i, name in enumerate(FAENAS)}
FAILURE_TYPE_TO_IDX = {name: i for i, name in enumerate(FAILURE_TYPES)}
RUL_SCALE_HOURS = 1000.0  # normaliza el target de RUL para balancear ambas perdidas

MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 300
PATIENCE = 15
VAL_FRACTION = 0.15  # fraccion de los equipos de train reservada para validacion


def _fingerprint(*tensors: torch.Tensor, **config) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(np.ascontiguousarray(t.numpy(), dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU == Swish (x * sigmoid(x))
}


def _build_activation(name: str) -> type[nn.Module]:
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Activacion desconocida: {name!r}. Opciones: {list(ACTIVATIONS)}") from exc


class MultiTaskDegradationNet(nn.Module):
    """Tronco compartido + cabeza de regresion (RUL) y cabeza de clasificacion (tipo de falla)."""

    def __init__(
        self,
        n_numeric_features: int,
        n_equipment_types: int = len(EQUIPMENT_TYPES),
        n_faenas: int = len(FAENAS),
        n_failure_classes: int = len(FAILURE_TYPES),
        embedding_dim: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.equipment_type_emb = nn.Embedding(n_equipment_types, embedding_dim)
        self.faena_emb = nn.Embedding(n_faenas, embedding_dim)

        act_layer = _build_activation(activation)
        trunk_input_dim = n_numeric_features + 2 * embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            act_layer(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_layer(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.rul_head = nn.Linear(hidden_dim, 1)
        self.failure_head = nn.Linear(hidden_dim, n_failure_classes)

    def forward(
        self, x_numeric: torch.Tensor, equipment_type_idx: torch.Tensor, faena_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.cat([self.equipment_type_emb(equipment_type_idx), self.faena_emb(faena_idx)], dim=1)
        shared = self.trunk(torch.cat([x_numeric, emb], dim=1))
        rul_pred = self.rul_head(shared).squeeze(-1)
        failure_logits = self.failure_head(shared)
        return rul_pred, failure_logits


def build_multi_task_table(
    features: pl.DataFrame, equipment_metadata: pl.DataFrame, maintenance_logs: pl.DataFrame
) -> pl.DataFrame:
    """Filas de telemetria de equipos fallados con ambas etiquetas: `rul_hours` y `failure_type`.

    Reutiliza `build_rul_training_table` y le suma el tipo de falla real
    (conocido retrospectivamente para cualquier lectura de un equipo que
    efectivamente fallo), habilitando entrenamiento multi-task por fila en
    vez de solo en la ultima lectura de cada equipo.
    """
    rul_df = build_rul_training_table(features, equipment_metadata)
    failure_events = maintenance_logs.filter(pl.col("event_type") == "falla_no_planificada").select(
        ["equipment_id", "failure_type"]
    )
    return rul_df.join(failure_events, on="equipment_id", how="inner")


def _map_to_idx(series: pl.Series, mapping: dict[str, int]) -> np.ndarray:
    return np.array([mapping[v] for v in series.to_list()], dtype=np.int64)


def _to_tensors(df: pl.DataFrame, scaler: StandardScaler, fit_scaler: bool) -> dict[str, torch.Tensor]:
    numeric = df.select(NUMERIC_COLUMNS).to_pandas().to_numpy(dtype=np.float32)
    numeric = scaler.fit_transform(numeric) if fit_scaler else scaler.transform(numeric)

    return {
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "equipment_type_idx": torch.tensor(_map_to_idx(df["equipment_type"], EQUIPMENT_TYPE_TO_IDX)),
        "faena_idx": torch.tensor(_map_to_idx(df["faena"], FAENA_TO_IDX)),
        "failure_idx": torch.tensor(_map_to_idx(df["failure_type"], FAILURE_TYPE_TO_IDX)),
        "rul_scaled": torch.tensor((df["rul_hours"].to_numpy() / RUL_SCALE_HOURS).astype(np.float32)),
    }


def train_multi_task_model(
    mt_df: pl.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    epochs: int = MAX_EPOCHS,
    batch_size: int = 256,
    lr: float = 1e-3,
    classification_loss_weight: float = 1.0,
    seed: int = 42,
    activation: str = "relu",
    patience: int = PATIENCE,
    min_epochs: int = MIN_EPOCHS,
    checkpoint_path: Path | None = None,
) -> tuple[MultiTaskDegradationNet, StandardScaler, dict]:
    """Entrena la red multi-task. La validacion para early stopping es un
    ``VAL_FRACTION`` de los *equipos* de train (todas sus lecturas juntas), nunca
    el test: antes se usaba el propio holdout de test para decidir cuando parar.
    Al menos ``min_epochs`` epocas (o ``epochs`` si es menor); desde ahi corta
    tras ``patience`` epocas sin mejora y restaura la mejor epoca. Con
    ``checkpoint_path`` el estado completo se guarda cada epoca y una corrida
    interrumpida retoma donde quedo."""
    torch.manual_seed(seed)

    rng = np.random.default_rng(seed)
    ordered_train = sorted(train_ids)
    n_val = max(1, int(round(len(ordered_train) * VAL_FRACTION)))
    val_ids = set(rng.choice(ordered_train, size=n_val, replace=False).tolist())
    fit_ids = set(ordered_train) - val_ids

    train_df = mt_df.filter(pl.col("equipment_id").is_in(fit_ids))
    val_df = mt_df.filter(pl.col("equipment_id").is_in(val_ids))
    test_df = mt_df.filter(pl.col("equipment_id").is_in(test_ids))

    scaler = StandardScaler()
    train_t = _to_tensors(train_df, scaler, fit_scaler=True)
    val_t = _to_tensors(val_df, scaler, fit_scaler=False)
    test_t = _to_tensors(test_df, scaler, fit_scaler=False)

    model = MultiTaskDegradationNet(n_numeric_features=len(NUMERIC_COLUMNS), activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        train_t["numeric"], train_t["equipment_type_idx"], train_t["faena_idx"], train_t["failure_idx"], train_t["rul_scaled"]
    )
    # drop_last evita que un ultimo batch de tamano 1 rompa BatchNorm en modo train.
    generator = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
    n_batches = len(train_loader)

    fp = _fingerprint(train_t["numeric"], val_t["numeric"], activation=activation, epochs=epochs,
                      batch_size=batch_size, lr=lr, w=classification_loss_weight, seed=seed,
                      patience=patience, min_epochs=min_epochs)
    state = {
        "epoch": 0,
        "loss_history": {"epoch": [], "train_total": [], "train_rul_mse": [], "train_cls_ce": [],
                         "val_total": [], "val_rul_mse": [], "val_cls_ce": []},
        "best_val": float("inf"), "best_epoch": 0, "best_state": None, "bad_epochs": 0, "done": False,
    }
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]
            print(f"  retomando desde el checkpoint en la epoca {state['epoch']}")

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    loss_history = state["loss_history"]
    while not state["done"] and state["epoch"] < epochs:
        generator.manual_seed(seed + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(seed + 10_000 + state["epoch"])
        model.train()
        epoch_total, epoch_rul, epoch_cls = 0.0, 0.0, 0.0
        for numeric, eq_idx, faena_idx, failure_idx, rul_scaled in train_loader:
            optimizer.zero_grad()
            rul_pred, failure_logits = model(numeric, eq_idx, faena_idx)
            rul_component = mse_loss(rul_pred, rul_scaled)
            cls_component = ce_loss(failure_logits, failure_idx)
            loss = rul_component + classification_loss_weight * cls_component
            loss.backward()
            optimizer.step()
            epoch_total += loss.item()
            epoch_rul += rul_component.item()
            epoch_cls += cls_component.item()
        epoch_total /= max(n_batches, 1)
        epoch_rul /= max(n_batches, 1)
        epoch_cls /= max(n_batches, 1)
        state["epoch"] += 1
        epoch = state["epoch"]

        model.eval()
        with torch.no_grad():
            val_rul_pred, val_failure_logits = model(val_t["numeric"], val_t["equipment_type_idx"], val_t["faena_idx"])
            val_rul_component = mse_loss(val_rul_pred, val_t["rul_scaled"]).item()
            val_cls_component = ce_loss(val_failure_logits, val_t["failure_idx"]).item()
            val_total = val_rul_component + classification_loss_weight * val_cls_component

        loss_history["epoch"].append(epoch)
        loss_history["train_total"].append(epoch_total)
        loss_history["train_rul_mse"].append(epoch_rul)
        loss_history["train_cls_ce"].append(epoch_cls)
        loss_history["val_total"].append(val_total)
        loss_history["val_rul_mse"].append(val_rul_component)
        loss_history["val_cls_ce"].append(val_cls_component)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch}/{epochs} - loss train: {epoch_total:.4f} | loss val: {val_total:.4f}")

        if val_total < state["best_val"] - 1e-7:
            state.update(best_val=val_total, best_epoch=epoch, bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if epoch >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
            print(f"  early stopping en epoch {epoch} (sin mejora en val_total por {patience} epochs; "
                  f"mejor: epoch {state['best_epoch']})")
        save_checkpoint()

    best_epoch = state["best_epoch"]
    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()

    model.eval()
    with torch.no_grad():
        rul_pred_scaled, failure_logits = model(test_t["numeric"], test_t["equipment_type_idx"], test_t["faena_idx"])
        rul_pred_hours = (rul_pred_scaled * RUL_SCALE_HOURS).numpy()
        rul_true_hours = (test_t["rul_scaled"] * RUL_SCALE_HOURS).numpy()
        failure_pred_idx = failure_logits.argmax(dim=1).numpy()
        failure_true_idx = test_t["failure_idx"].numpy()

    metrics = {
        "mae_rul_hours": float(mean_absolute_error(rul_true_hours, rul_pred_hours)),
        "accuracy_failure_type": float(accuracy_score(failure_true_idx, failure_pred_idx)),
        "f1_macro_failure_type": float(f1_score(failure_true_idx, failure_pred_idx, average="macro")),
        "n_train_rows": train_df.height,
        "n_val_rows": val_df.height,
        "n_val_equipment": len(val_ids),
        "n_test_rows": test_df.height,
        "activation": activation,
        "loss_history": loss_history,
        "best_epoch": best_epoch,
        "early_stopped": len(loss_history["epoch"]) < epochs,
        "patience": patience,
    }
    return model, scaler, metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    equipment_metadata = pl.read_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    maintenance_logs = pl.read_parquet(PROCESSED_DIR / "maintenance_logs.parquet")
    features = pl.read_parquet(PROCESSED_DIR / "telemetry_features.parquet")

    mt_df = build_multi_task_table(features, equipment_metadata, maintenance_logs)

    # Mismo criterio de split (semilla + equipos fallados) que train_survival_pipeline,
    # para que la comparacion de MAE de RUL sea sobre el mismo holdout.
    failed_ids = mt_df["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.25, seed=42)

    print(f"Entrenando red multi-task (PyTorch) sobre {mt_df.height} lecturas de {len(failed_ids)} equipos fallados...")
    model, scaler, metrics = train_multi_task_model(
        mt_df, train_ids, test_ids, checkpoint_path=MODELS_DIR / "checkpoints" / "multi_task_relu.ckpt"
    )

    print(
        f"\n[Multi-task PyTorch] Early stopping: mejor epoch {metrics['best_epoch']}"
        f"{f' (detenido antes de las {MAX_EPOCHS})' if metrics['early_stopped'] else f' (corrio las {MAX_EPOCHS} completas)'}"
    )
    print(f"[Multi-task PyTorch] MAE RUL: {metrics['mae_rul_hours']:.1f} horas")
    print(
        f"[Multi-task PyTorch] Clasificacion de falla (por lectura, no solo la ultima): "
        f"Accuracy {metrics['accuracy_failure_type']:.3f} | F1-macro {metrics['f1_macro_failure_type']:.3f}"
    )

    torch.save(model.state_dict(), MODELS_DIR / "multi_task_net.pt")
    joblib.dump(scaler, MODELS_DIR / "multi_task_scaler.joblib")

    metrics_path = PROCESSED_DIR / "metrics.json"
    existing = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing["multi_task_pytorch"] = metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en: {MODELS_DIR / 'multi_task_net.pt'}")
    print(f"Metricas combinadas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
