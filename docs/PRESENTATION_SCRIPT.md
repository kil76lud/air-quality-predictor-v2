# Guion de exposición — 4 integrantes, 14:30 min (límite 15:00)

| # | Integrante | Tiempo | Rúbrica | Contenido |
|---|---|---|---|---|
| 1 | **A · Producto y app** | 0:00-3:00 (3:00) | App en inglés (2) | Problema (calidad del aire en Puno, por qué t+6h); arquitectura en 1 diapositiva; **demo de la app** en inglés: *Live* (/latest) y modo manual; `/model-info` |
| 2 | **B · Datos y modelo** | 3:00-7:00 (4:00) | Dataset/entrenamiento/features/hiperparámetros + métricas (4) | Open-Meteo (CAMS, limitación de cuadrícula); 12 meses + respaldo; features (lags 1/3/6/24, medias 6/24); **no-leakage** (target `shift(-6)`, split cronológico, test); distribución de clases; baseline vs XGBoost+Optuna (30 trials, TimeSeriesSplit); F1 macro, matriz de confusión, importancias; MLflow (alias champion) |
| 3 | **C · Despliegue + CI** | 7:00-11:00 (4:00) | Despliegue (2) + CI (2 de 4) | Docker (imagen ligera, healthcheck) y HF Spaces; secrets; montaje de la UI. **En vivo:** `git push` → CI (ruff, pytest, quality gate, docker build); mostrar un CI rojo por quality gate y su corrección |
| 4 | **D · Retrain, CD y cierre** | 11:00-14:30 (3:30) | CI/CD (2 de 4) + mantenimiento | **En vivo:** Retrain → challenger **rechazado** y **promovido**; CD redespliega; `version` cambia en `/model-info`. Runbook en 20 s; conclusiones y trabajo futuro (más ciudades, sensores locales, monitoreo de deriva) |

## Guía por integrante
**A (3:00)** — 0:00 saludo y problema (0:30) · 0:30 arquitectura (0:30) · 1:00 demo app en inglés (1:45) · 2:45 puente a B.
**B (4:00)** — datos y respaldo (0:45) · features y leakage (1:15) · distribución de clases y baseline vs optimizado (1:00) · Optuna + importancias + MLflow (1:00).
**C (4:00)** — contenedor y Render (1:00) · pipeline CI explicado (0:45) · push en vivo + resultado (1:30) · CI rojo/verde (0:45).
**D (3:30)** — retrain manual rechazado (0:50) · promovido + redeploy (1:30) · runbook (0:20) · conclusiones (0:50).

## Reglas de equipo
- Un ensayo cronometrado completo antes de la entrega (objetivo 14:00).
- Plan B por cada paso en vivo (capturas/grabación de 30 s) — `docs/DEMO_SCRIPT.md`.
- Hablar de la promoción con `min_delta=-1.0` como **umbral de demostración**.
- Todos deben poder responder: ¿por qué no hay leakage?, ¿qué pasa si la API cae?, ¿cómo se revierte un champion?
