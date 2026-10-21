"""Diagnostico: MAE del MLP en el banco no visto segun cuantas epocas entrena.

Entrena con un numero fijo de epocas (``min_epochs = epochs``, asi el early
stopping no corta antes) y evalua con el mismo GroupKFold
leave-one-experiment-out que el resto del proyecto. Es un diagnostico, no un
criterio de seleccion: elegir las epocas con esta tabla seria elegir mirando
el banco de test.

    python -m src.dl_epoch_sweep
"""

from __future__ import annotations

import json

from src.dl_model import evaluate_activations_group_kfold
from src.dl_pipeline import REPORTS_DIR, STRIDE
from src.features import build_feature_table
from src.ingest import load_all_snapshots

EPOCHS = (80, 200)


def main() -> None:
    table = build_feature_table(load_all_snapshots(stride=STRIDE))
    feature_cols = [c for c in table.columns if c not in ("experiment", "file_index", "time_to_failure_min", "rul_fraction")]
    out = {}
    for epochs in EPOCHS:
        results = evaluate_activations_group_kfold(
            table[feature_cols], table["rul_fraction"].to_numpy(), table["experiment"].to_numpy(),
            epochs=epochs, min_epochs=epochs,
        )
        out[epochs] = results.to_dict(orient="records")
        print(f"\n=== {epochs} epocas fijas ===")
        print(results.to_string(index=False))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dl_epoch_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
