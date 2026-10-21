# Air Quality Category Predictor

Predice la categoría **US AQI** (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy,
Very Unhealthy) **6 horas en el futuro** para **Puno, Perú** (-15.8375, -70.02167) con datos de
la Open-Meteo Air Quality API. Proyecto DevOps + MLOps: ingesta validada, entrenamiento
reproducible (XGBoost + Optuna + MLflow), champion/challenger, CI/CD con GitHub Actions y
despliegue en Render (Docker, plan gratuito).

> ⚠️ **El repositorio se entrega con un modelo PLACEHOLDER entrenado con datos sintéticos**
> (`data_source: "synthetic"` en `models/champion/metadata.json`) para que Docker, CI y la API
> funcionen desde el primer minuto. **Antes de la demo ejecuta `make first-run`**: descarga el
> histórico REAL, reconstruye el dataset y entrena el primer champion real.

## Inicio rápido

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install
make test                 # 58 pruebas (sin red)
make first-run            # datos reales + primer champion de CADA ciudad (≈ 3-6 min por ciudad)
make serve                # http://localhost:7860  (UI) · /docs (Swagger)
make mlflow-ui            # http://localhost:5000
```

Docker: `docker compose up --build`  ·  MLflow en Docker: `docker compose --profile mlops up`

## Multiciudad (selector en la app)

El catálogo vive en `config/config.yaml` (`city` = ciudad por defecto, `cities` = el resto). Cada
ciudad tiene **su propio** dataset, champion y reportes: `data/raw/<slug>.parquet`,
`data/processed/<slug>.parquet`, `models/champion/<slug>/`, `models/reports/<slug>/`, y un modelo
registrado por ciudad en MLflow (`aqi-category-classifier-<slug>`, alias `champion`).

```bash
python -m aqi.download --all            # o --city lima
python -m aqi.ingest --reset --all
python -m aqi.train --force-promote --all
```

API: `GET /cities` (catálogo + `available`) y el parámetro opcional `?city=<slug>` en
`/model-info`, `/predict` y `/latest` (sin parámetro = ciudad por defecto). Una ciudad sin
champion responde 503; un slug desconocido, 404. `retrain.yml` reentrena todas las ciudades.

## Mapa del repositorio

| Ruta | Responsable | Contenido |
|---|---|---|
| `aqi/` | MLOps | datos, features, modelo, champion, API |
| `app/ui.py` | **Diseñador UI** | `build_ui() -> gr.Blocks` (hoy placeholder) |
| `main.py` | MLOps | monta la UI sobre FastAPI (`gr.mount_gradio_app`) |
| `config/config.yaml` | MLOps | ciudad, horizonte, trials, umbrales |
| `data/raw` · `data/processed` | MLOps | bootstrap inmutable (respaldo) · dataset acumulado |
| `models/champion` · `models/reports` | MLOps | modelo en producción · figuras y bitácora de decisiones |
| `tests/` | MLOps | ingesta, features, **no-leakage**, API, champion, **quality gate** |
| `.github/workflows` | MLOps | `ci.yml`, `retrain.yml`, `cd.yml` |
| `docs/` | Equipo | despliegue, demo, informe, runbook, exposición, plan |

## Contrato de la API (compartido con la UI; no cambiar sin avisar)

`GET /health` · `GET /model-info` · `POST /predict` · `GET /latest`. Errores: `{"detail": str}`.
Esquemas en `aqi/schemas.py`; ejemplos interactivos en `/docs`.

**Modo manual (`/predict`)**: el usuario solo aporta el estado actual; los **lags y promedios
móviles se rellenan con el valor actual** de cada contaminante (`aqi/features.py::row_from_current`).
Esto es una aproximación (atmósfera estable); `/latest` usa el historial real de las últimas 72 h.
`day_of_week`: lunes = 0 … domingo = 6. `hour`: hora local de Puno (UTC-5).

## Decisiones clave

* **Anti-leakage**: target = categoría de `us_aqi` en t+6 (`shift(-6)`); features solo con
  información ≤ t; `us_aqi` jamás es feature; split cronológico con hueco = horizonte;
  `TimeSeriesSplit(gap=6)`. Verificado en `tests/test_no_leakage.py`.
* **Clima**: la Air Quality API no trae clima → se combina con la API meteorológica
  (archive + forecast `past_days`).
* **Categorías**: todo AQI > 200 (incl. "Hazardous") se agrupa en *Very Unhealthy*.
* **Clases ausentes**: en una ciudad de baja contaminación algunas clases pueden no existir;
  `AqiClassifier` siempre devuelve 5 probabilidades y el F1 macro se promedia sobre las
  clases presentes en el test.
* **Limitación**: datos de **reanálisis por cuadrícula (CAMS)**, no de sensores locales; la
  celda cubre decenas de km y no captura microclimas urbanos ni picos locales.
* **Champion/challenger**: mismo test para ambos; se promueve si ΔF1 macro ≥ +0.005.
* **Reproducibilidad**: semillas fijas, dataset y champion versionados en git, dependencias fijadas.
