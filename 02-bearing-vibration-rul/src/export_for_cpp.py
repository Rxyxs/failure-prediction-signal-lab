"""Exporta 5 snapshots de vibracion reales (crudos, primeros 16384 puntos --
la potencia de 2 mas cercana a los 20.480 puntos originales, para permitir
una FFT radix-2 sin dependencias en cpp/) mas las features de referencia
calculadas por Python sobre ese mismo recorte exacto, para verificacion
bit-a-bit contra cpp/fft_features.

    python -m src.export_for_cpp
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ingest import EXPERIMENTS
from src.features import extract_features

ROOT = Path(__file__).resolve().parent.parent
CPP_DIR = ROOT / "cpp"
TRUNCATED_LENGTH = 16384  # 2^14, la potencia de 2 mas cercana a 20480

FEATURE_ORDER = ["rms", "kurtosis", "dominant_frequency_hz", "spectral_entropy", "energy_2000_5000hz"]


def main() -> None:
    folder = EXPERIMENTS["2nd_test"]
    files = sorted(folder.iterdir(), key=lambda p: p.name)
    sample_files = [files[0], files[len(files) // 2], files[-1], files[10], files[-10]]

    CPP_DIR.mkdir(parents=True, exist_ok=True)
    ref_lines = ["file," + ",".join(FEATURE_ORDER)]

    for f in sample_files:
        arr = np.loadtxt(f)
        channel = arr[:, np.argmax(arr.std(axis=0))] if arr.ndim > 1 else arr
        truncated = channel[:TRUNCATED_LENGTH]

        raw_path = CPP_DIR / f"snapshot_{f.name}.txt"
        np.savetxt(raw_path, truncated, fmt="%.6f")

        feats = extract_features(truncated)
        values = [str(feats[k]) for k in FEATURE_ORDER]
        ref_lines.append(f"{f.name}," + ",".join(values))
        print(f"{f.name}: " + ", ".join(f"{k}={feats[k]:.6f}" for k in FEATURE_ORDER))

    (CPP_DIR / "python_reference.csv").write_text("\n".join(ref_lines), encoding="utf-8")
    print(f"\nGuardado en: {CPP_DIR}/snapshot_*.txt, python_reference.csv")


if __name__ == "__main__":
    main()
