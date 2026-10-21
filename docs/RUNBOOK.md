# Runbook operativo

| Símbolo | Significado |
|---|---|
| 🔎 | cómo detectarlo |
| 🛠️ | qué hacer |
| ↩️ | cómo revertir |

## 1. Falla de ingesta
**Síntomas**: `retrain.yml` termina con `INGEST REJECTED (invalid data)` (exit 2) o `INGEST FAILED (API)` (exit 3); o el log dice `[fallback] Open-Meteo unavailable...`.
- **API caída / timeout (modo `fallback`)** 🔎 log `status=fallback`. 🛠️ El retrain continúa con el dataset versionado (no falla). Reintentar con *Run workflow* más tarde. Si se exige datos frescos: `python -m aqi.ingest --strict`.
- **Datos inválidos (pandera)** 🔎 mensaje con tabla `failure_cases` (columna, check, valor). 🛠️ Nada se escribe en `data/processed`. Revisar si Open-Meteo cambió unidades/variables (`aqi/download.py::AIR_VARS/WEATHER_VARS`) o si es un valor aislado; ajustar rangos en `aqi/schemas.py` solo si el valor es físicamente posible.
- **Dataset corrupto** ↩️ `git checkout -- data/processed/dataset.parquet` o reconstruir: `python -m aqi.ingest --reset` (parte del bootstrap inmutable `data/raw`).
- **Bootstrap imposible** (API caída el día de la entrega) ↩️ usar el `data/raw/bootstrap.parquet` ya versionado.

## 2. Degradación del modelo
**Síntomas**: CI rojo en `test_champion_reevaluated_on_current_holdout_above_minimum` ("Champion degraded"); F1 macro del champion cae semana a semana en `decisions.jsonl` (`champion_f1_macro`); quejas de la UI.
🛠️ Pasos:
1. `cat models/reports/decisions.jsonl | tail` → tendencia de `champion_f1_macro` y `challenger_f1_macro`.
2. Ver `models/reports/class_distribution.png` (¿cambió el régimen? p. ej. temporada de quemas) y `confusion_challenger.png`.
3. Reentrenar con más búsqueda: *Run workflow* con `n_trials=80`.
4. Si ningún challenger supera al champion pero el champion bajó del mínimo: revisar features/datos (deriva) y, como medida temporal, bajar `quality_gate_f1_macro` **documentando el motivo**.
↩️ Volver al champion anterior: `git revert <commit chore(retrain): promote...>` → push → CI → CD. (La versión previa vuelve a `models/champion/`; en MLflow reasignar el alias: *Models → aqi-category-classifier → versión anterior → Add alias "champion"*.)

## 3. Falla de despliegue
**Síntomas**: `cd.yml` rojo; deploy en Render en *Failed*; `/health` no responde.
- **`Falta el secret ...`** 🛠️ Configurar `RENDER_DEPLOY_HOOK_URL` y `RENDER_APP_URL` (docs/DEPLOY.md §4).
- **Deploy Hook responde 401/404** 🛠️ URL regenerada o mal copiada: Render → *Settings → Deploy Hook* y actualizar el secret `RENDER_DEPLOY_HOOK_URL`.
- **CD espera 15 min y falla ("la versión nueva no apareció")** 🔎 Logs del servicio en Render (¿build fallido?). Si el deploy ya está *Live* pero CD falló, el commit desplegado difiere del esperado (llegó otro push): revisar `X-App-Commit` con `curl -si <url>/health`.
- **Primera respuesta lenta (≈ 1 min)** 🔎 El plan gratuito de Render duerme el servicio tras 15 min sin tráfico: es esperado; abrir la app 10 min antes de exponer.
- **Build error en Render** 🔎 *Logs* del servicio. Causas típicas: dependencia sin versión fija, falta `models/champion` (¿se versionó?), poca memoria en el build. 🛠️ Reproducir en local: `docker build -t aqi-app . && docker run -p 7860:7860 aqi-app`.
- **Runtime error / UI no carga** 🔎 *Logs* del servicio (runtime). `main.py` mantiene la API viva aunque falle la UI (`UI could not be mounted`): corregir `app/ui.py` y hacer push.
- **`/health` ok pero `/model-info` 503** 🛠️ Falta `models/champion/<ciudad>/model.joblib` o `metadata.json` en la imagen (la ciudad aún no se entrenó).
- **`/latest` 503** 🛠️ Open-Meteo no responde y no hay `last_window.parquet`; la API ya intenta ese respaldo automáticamente.
↩️ Rollback: `git revert <commit>` y push (CD redespliega), o *Actions → CD → Run workflow* sobre un commit estable. En Render también puede volverse a un deploy anterior desde el panel del servicio.

## 4. Mantenimiento rutinario
- Semanal (automático): `retrain.yml` (lunes 06:00 UTC).
- Mensual: revisar `decisions.jsonl`, deriva de clases y tamaño del repo (`mlflow.db`, `data/processed`).
- Rotar el Deploy Hook (*Regenerate* en Render) cada 90 días y actualizar el secret.
