[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Failure Prediction from Signal Lab

El mismo problema de fondo — predecir tiempo hasta falla desde una señal continua — aplicado a tres dominios y tipos de señal distintos: datos operacionales de flota, datos de sensores de vibración/acústicos, y señal sísmica acústica continua. Cada carpeta es autocontenida, con su propio README, dependencias y tests. Este repo reemplaza tres repos separados de un solo dominio que antes vivían en este perfil.

## Técnicas

| # | Dominio | Carpeta | Qué hace |
|---|---|---|---|
| 01 | Flota minera (multi-task) | [`01-mining-fleet-multitask-rul`](01-mining-fleet-multitask-rul) | Modelo de supervivencia CoxPH + LightGBM + una red multi-task en PyTorch + SHAP predicen la vida útil remanente (RUL) de camiones CAEX sobre datos sintéticos realistas de flota, servido vía FastAPI/Streamlit. |
| 02 | Vibración de rodamientos (datos reales) | [`02-bearing-vibration-rul`](02-bearing-vibration-rul) | Ingeniería de features FFT/espectrales + LightGBM/CatBoost predicen el tiempo remanente hasta falla sobre el dataset real NASA IMS Bearing, con un componente de procesamiento de señal en C++. |
| 03 | Señal sísmica acústica | [`03-earthquake-acoustic-signal`](03-earthquake-acoustic-signal) | Features FFT/espectrales + LightGBM/CatBoost predicen `time_to_failure` desde señal acústica sísmica continua (LANL/Kaggle), servido vía DuckDB + CLI. |

## Qué encontró cada uno de los tres dominios

Todos los números provienen de una corrida real del pipeline de esa carpeta. Los tres comparten una tesis que la tabla hace visible:

| # | Dominio | Número principal | Qué dice en realidad |
|---|---|---|---|
| **01** | Flota minera (sintético) | C-Index de CoxPH **0,6246**; MAE de RUL **513 h** sobre ciclos de hasta ~16.700 h | Semanas de anticipación, no una alarma de último minuto. El modelo de supervivencia rankea toda la flota **incluyendo unidades que todavía no fallaron** (censuradas), algo que las predicciones puntuales sobre unidades falladas no pueden hacer |
| **01** | — misma carpeta | Accuracy de tipo de falla **1,000** en la última lectura, **0,744** sobre todas las lecturas | El mismo clasificador, dos protocolos de evaluación, 0,26 de diferencia. Puntuar solo la lectura final antes de la falla es la pregunta fácil; puntuar cada lectura del ciclo es la que enfrenta un operador |
| **02** | Vibración de rodamientos (**real**, NASA IMS) | MAE **14.776 min → 0,216** de vida restante | No es un triunfo de modelado — es un triunfo de **definición del target**. Ver abajo |
| **02** | — misma carpeta | Optuna, 30 trials: 0,2159 → **0,2080** (3,7%) | Declarado modesto a propósito: 3 folds de GroupKFold, uno por banco físico, es una señal delgada para búsqueda de hiperparámetros |
| **03** | Acústica sísmica (LANL) | MAE del ensemble NNLS **1,474**; CNN 1D sobre señal cruda **~4,5** | El gradient boosting sobre features espectrales construidas a mano le gana **2,7x** a una CNN que lee la señal cruda. La ingeniería de features *es* el modelo |

---

## Evidencia

### La definición del target le ganó a toda elección de modelo

![MAE antes y después de corregir el target](02-bearing-vibration-rul/outputs/reports/target_fix_comparison.png)

**Cómo leerla — los dos paneles no comparten unidades.** A la izquierda el MAE en minutos absolutos, a la derecha el MAE como fracción de vida restante [0,1]. No son comparables barra a barra; lo que importa es que el panel izquierdo es un fracaso y el derecho un modelo que funciona.

El primer intento predecía minutos crudos hasta la falla y producía errores de **miles de minutos** en los folds leave-one-experiment-out. La causa no era el modelo: los tres experimentos de NASA IMS duran plazos radicalmente distintos — aproximadamente 15, 7 y 44 días — así que un modelo entrenado con dos nunca vio la escala temporal absoluta del tercero. Cambiar el target a **RUL fraccional** (`snapshots_restantes / snapshots_totales`), la corrección estándar en la literatura de RUL por exactamente este motivo, volvió tratable el problema.

Dos detalles que vale notar. Primero, la mejora es de **3 a 8x según el modelo** — muchísimo más que el 3,7% que aportaron encima 30 trials de Optuna. Segundo, **cambia el propio ranking de modelos**: Random Forest es el mejor de los cinco con el target roto y apenas cuarto con el corregido. Un benchmark corrido sobre un target mal especificado habría elegido el modelo equivocado y lo habría reportado con confianza.

### Las features construidas a mano le ganan a la CNN sobre la señal cruda

![MAE out-of-fold por modelo](03-earthquake-acoustic-signal/reports/figures/mae_comparison.png)

**Cómo leerla.** MAE out-of-fold sobre `time_to_failure`, más bajo es mejor. Seis modelos individuales más el ensemble ponderado por NNLS en violeta. La CNN 1D (`cnn_1d`) es el único modelo que consume la **señal acústica cruda**; todos los demás ven features FFT y estadísticas calculadas sobre esa misma señal.

La CNN queda anteúltima en ~4,5, superada 2,7x por CatBoost en 1,650 sobre features construidas. Sobre una señal sísmica continua con esta cantidad de datos por segmento, decidir *qué medir* — bandas de energía espectral, estadísticas rodantes, percentiles — cargó muchísima más información que dejar que una red convolucional la descubriera. El ensemble en 1,474 mejora después un 11% adicional sobre el mejor modelo individual, con NNLS eligiendo los pesos en vez de promediar a ciegas.

Notar también la distancia de los modelos lineales: Ridge en 3,973 y Lasso en 4,621 no se acercan a los ensembles de árboles. La relación entre features espectrales y tiempo hasta la falla no es lineal, y las líneas base lineales se mantienen en la tabla para mostrarlo en vez de afirmarlo.

---

## El patrón que cruza los tres

Tres dominios — telemetría de flota minera, vibración de rodamientos, acústica sísmica — y una conclusión compartida:

> **Las decisiones de encuadre le ganan a la elección de modelo, siempre.**

- **El target** (02): redefinirlo valió 3–8x. La mejor búsqueda de hiperparámetros encima del target correcto valió 3,7%.
- **Las features** (03): la ingeniería espectral sobre la señal cruda le ganó 2,7x a una CNN que leía esa misma señal directamente.
- **El protocolo de evaluación** (01): el mismo clasificador de tipo de falla reporta 1,000 o 0,744 según si se puntúa la última lectura o todas.

Esa es también la razón por la que los tres viven en un solo repo. Los dominios suenan inconexos — camiones de extracción, rodamientos, terremotos — pero el toolkit es el mismo: ingeniería de features espectrales y estadísticas alimentando modelos de supervivencia y regresión con gradient boosting, validados con un agrupamiento que respeta la independencia física de cada banco, corrida o unidad.

---

## Por qué un repo en vez de tres

Cada técnica es real, ejecutable y probada de forma independiente — esto no es esconder alcance, es representarlo con precisión. Tres repos en tres dominios que suenan sin relación (minería, rodamientos, terremotos) esconden el hecho de que comparten la misma técnica central (ingeniería de features espectrales/estadísticas alimentando modelos de supervivencia/regresión con gradient boosting); un laboratorio hace de esa transferibilidad el punto real — el mismo toolkit aplicado a operaciones mineras, ingeniería mecánica y geofísica.

## Cómo correr una técnica

Cada carpeta es autocontenida — ver su propio README para el setup exacto y el entry point, resultados reales de una corrida real, y cualquier hallazgo negativo honesto.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
