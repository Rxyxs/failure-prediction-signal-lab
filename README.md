[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Failure Prediction from Signal Lab

The same underlying problem — predicting time-to-failure from a continuous signal — applied to three different domains and signal types: fleet operational data, vibration/acoustic sensor data, and continuous seismic acoustic data. Each folder is self-contained with its own README, dependencies, and tests. This repo replaces three separate single-domain repos that used to live on this profile.

## Techniques

| # | Domain | Folder | What it does |
|---|---|---|---|
| 01 | Mining fleet (multi-task) | [`01-mining-fleet-multitask-rul`](01-mining-fleet-multitask-rul) | CoxPH survival model + LightGBM + a PyTorch multi-task network + SHAP predict remaining useful life (RUL) for CAEX trucks from synthetic realistic fleet data, served via FastAPI/Streamlit. |
| 02 | Bearing vibration (real data) | [`02-bearing-vibration-rul`](02-bearing-vibration-rul) | FFT + spectral feature engineering + LightGBM/CatBoost predict remaining time-to-failure from the real NASA IMS Bearing Dataset, with a C++ signal-processing component. |
| 03 | Seismic acoustic signal | [`03-earthquake-acoustic-signal`](03-earthquake-acoustic-signal) | FFT + spectral features + LightGBM/CatBoost predict `time_to_failure` from continuous acoustic seismic data (LANL/Kaggle), served via DuckDB + CLI. |

## Why one repo instead of three

Each technique is real, runnable, and independently tested — this isn't about hiding scope, it's about representing it accurately. Three repos in three unrelated-sounding domains (mining, bearings, earthquakes) hide the fact that they share the same core technique (spectral/statistical feature engineering into gradient-boosted survival/regression models); one lab makes that transferability the actual point — the same toolkit applied across mining operations, mechanical engineering, and geophysics.

## Running a technique

Each folder is self-contained — see its own README for the exact setup and entry point, real results from an actual run, and any honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
