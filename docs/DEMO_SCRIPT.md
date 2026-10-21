# Guion de demo en vivo (≈ 6-7 min dentro de la exposición)

**Antes de empezar (checklist, 10 min antes):** servicio de Render *Live* y despierto (abrir la URL 10 min antes); pestañas abiertas: app, GitHub
Actions, MLflow UI (`make mlflow-ui`), terminal en el repo con el venv activo; `make first-run`
ya ejecutado; **Plan B**: capturas de cada paso en una carpeta.

## 1. App en inglés (1 min)
1. Abrir la URL de la app en Render. Mostrar *Live* (`/latest`): última observación de Puno y la categoría predicha a t+6h con probabilidades.
2. Modo manual: mover PM2.5 / humedad y pulsar *Predict*. Mencionar que lags y medias móviles se rellenan con el valor actual.
3. Abrir `/model-info` y `/docs`.

## 2. git push que dispara CI (1.5 min)
```bash
git checkout -b demo-ci
echo "" >> README.md && git commit -am "docs: demo push" && git push -u origin demo-ci
```
Actions → CI en curso: *ruff → pytest (ingesta, features, no-leakage, API, quality gate) → docker build + smoke test*.
**CI rojo (opcional, impactante):** en `config/config.yaml` poner `quality_gate_f1_macro: 0.99`,
commit y push → el job falla en `test_quality_gate`. Revertir (`git revert HEAD`) → verde.
Mostrar en un test (`tests/test_no_leakage.py`) qué verifica.

## 3. Retrain manual: challenger RECHAZADO (1.5 min)
Actions → **Retrain → Run workflow** con `min_delta = 0.5` (umbral exigente → imposible superarlo).
Mostrar el *Summary*: `⛔ RECHAZADO` + motivo (`delta +0.00xx < required +0.5000`). Mostrar
`models/reports/decisions.jsonl` y en MLflow el tag `decision=rejected` de esa versión.
*Alternativa local sin red de Actions:* `make demo-reject` (mismos datos/seed/config → challenger idéntico → rechazado).

## 4. Retrain manual: challenger PROMOVIDO + redeploy (2 min)
Actions → **Retrain → Run workflow** con `min_delta = -1.0` (**umbral de demostración**, decirlo en voz alta:
fuerza el camino de promoción para enseñar el mecanismo; en producción el umbral es +0.005).
Mostrar `✅ PROMOVIDO`, el commit `chore(retrain): promote new champion` y cómo se lanza **CD**.
Esperar el redeploy (≈ 2-4 min; mientras tanto explicar el champion/challenger) y refrescar `/model-info`:
**la `version` aumentó**. En MLflow: alias `champion` apunta a la nueva versión.

## 5. Cierre (30 s)
Mostrar en MLflow: historial de Optuna, importancia de features, baseline vs optimizado (o los PNG de `models/reports/`).

## Notas honestas para el Q&A
- La promoción con `-1.0` es de demostración; la regla real exige +0.005 de F1 macro en el mismo test.
- Un push hecho por `GITHUB_TOKEN` no dispara otros workflows, por eso `retrain.yml` llama a `cd.yml` con `gh workflow run`.
- Si GitHub/HF caen: ejecutar los mismos pasos en local (`make train`, `docker compose up`) y mostrar las capturas.
