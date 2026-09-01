[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Failure Prediction from Signal Lab

El mismo problema de fondo — predecir tiempo hasta falla desde una señal continua — aplicado a tres dominios y tipos de señal distintos: datos operacionales de flota, datos de sensores de vibración/acústicos, y señal sísmica acústica continua. Cada carpeta es autocontenida, con su propio README, dependencias y tests. Este repo reemplaza tres repos separados de un solo dominio que antes vivían en este perfil.

## Técnicas

| # | Dominio | Carpeta | Qué hace |
|---|---|---|---|
| 01 | Flota minera (multi-task) | [`01-mining-fleet-multitask-rul`](01-mining-fleet-multitask-rul) | Modelo de supervivencia CoxPH + LightGBM + una red multi-task en PyTorch + SHAP predicen la vida útil remanente (RUL) de camiones CAEX sobre datos sintéticos realistas de flota, servido vía FastAPI/Streamlit. |
| 02 | Vibración de rodamientos (datos reales) | [`02-bearing-vibration-rul`](02-bearing-vibration-rul) | Ingeniería de features FFT/espectrales + LightGBM/CatBoost predicen el tiempo remanente hasta falla sobre el dataset real NASA IMS Bearing, con un componente de procesamiento de señal en C++. |
| 03 | Señal sísmica acústica | [`03-earthquake-acoustic-signal`](03-earthquake-acoustic-signal) | Features FFT/espectrales + LightGBM/CatBoost predicen `time_to_failure` desde señal acústica sísmica continua (LANL/Kaggle), servido vía DuckDB + CLI. |

## Por qué un repo en vez de tres

Cada técnica es real, ejecutable y probada de forma independiente — esto no es esconder alcance, es representarlo con precisión. Tres repos en tres dominios que suenan sin relación (minería, rodamientos, terremotos) esconden el hecho de que comparten la misma técnica central (ingeniería de features espectrales/estadísticas alimentando modelos de supervivencia/regresión con gradient boosting); un laboratorio hace de esa transferibilidad el punto real — el mismo toolkit aplicado a operaciones mineras, ingeniería mecánica y geofísica.

## Cómo correr una técnica

Cada carpeta es autocontenida — ver su propio README para el setup exacto y el entry point, resultados reales de una corrida real, y cualquier hallazgo negativo honesto.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
