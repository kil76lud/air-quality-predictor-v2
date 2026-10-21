# Guía de despliegue (Render + GitHub Actions)

## 0. Requisitos
Python 3.11, Git, Docker (opcional en local), cuenta de GitHub y de Render (plan gratuito).

## 1. Datos reales y primer champion de cada ciudad (local, una sola vez)
```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
make install
make first-run          # download -> ingest --reset -> train --force-promote, para TODAS las ciudades
```
(Sin `make`, en Windows: `python -m aqi.download --all`, `python -m aqi.ingest --reset --all`,
`python -m aqi.train --force-promote --all`.)

Revisa en cada ciudad que `models/reports/<ciudad>/class_distribution.png` muestre varias clases y que
`models/champion/<ciudad>/metadata.json` tenga `"data_source": "open-meteo"`. Si Open-Meteo falla,
reintenta (`--city <slug>` repite solo una); el pipeline NO inventa datos reales.
Opcional: `make mlflow-ui` → experimento `air-quality-aqi`, un modelo registrado por ciudad
(`aqi-category-classifier-<ciudad>`) con alias **champion**.

## 2. Repositorio en GitHub
```bash
git init -b main && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/<usuario>/<repo>.git
git push -u origin main
```
`mlflow.db` **sí se versiona**; `mlartifacts/` no. En GitHub → **Settings → Actions → General →
Workflow permissions → "Read and write permissions"** (para que `retrain.yml` haga commit y dispare `cd.yml`).

## 3. Servicio web en Render (plan gratuito)
**Opción A: Blueprint (recomendada).** render.com → *New* → *Blueprint* → conecta el repo → Render lee
`render.yaml` (Docker, plan free, health check `/health`, auto-deploy desactivado) → *Apply*.

**Opción B: manual.** *New* → *Web Service* → repo → Language **Docker** → Instance type **Free** →
Health Check Path `/health` → Auto-Deploy **No / Off**. (Los nombres de campos pueden variar un poco.)

No definas la variable `PORT`: Render la inyecta (10000) y el contenedor la lee.

Luego, en el servicio: **Settings → Deploy Hook** → copia la URL (es secreta). Anota también la URL
pública, `https://<nombre>.onrender.com`.

## 4. Secrets de GitHub (Settings → Secrets and variables → Actions → *New repository secret*)
| Secret | Valor |
|---|---|
| `RENDER_DEPLOY_HOOK_URL` | URL completa del Deploy Hook (incluye `?key=...`) |
| `RENDER_APP_URL` | URL pública del servicio, p. ej. `https://air-quality-predictor.onrender.com` |

`GITHUB_TOKEN` lo provee GitHub automáticamente. Crea los secrets **antes** del primer push a `main`.

## 5. Primer despliegue
Un push a `main` ejecuta CI y, si queda verde, CD. O a mano: pestaña **Actions** → *CD* → **Run workflow**.
`cd.yml` dispara el Deploy Hook (Render construye el Dockerfile desde GitHub) y espera hasta que
`/health` responda con la cabecera `X-App-Commit` igual al commit desplegado (≈ 5-10 min la primera vez).
Verifica en PowerShell:
```powershell
$u = "https://<nombre>.onrender.com"
curl.exe -s "$u/health"; curl.exe -s "$u/cities"; curl.exe -s "$u/model-info?city=lima"; curl.exe -s "$u/latest?city=puno"
```

## 6. Interfaz
La UI vive en `app/web/index.html` (HTML + CSS + JS, sin build, selector de ciudad y botón EN/ES) y se
monta desde `app/ui.py` (`build_ui() -> gr.Blocks`) en la raíz `/`; la API queda en `/health`, `/cities`,
`/model-info`, `/predict`, `/latest` y `/docs`. Llama a la API con rutas relativas, así que funciona
igual en local y en Render. Prueba local: `make serve` → http://localhost:7860. Docker: `docker compose up --build`.

## 7. Plan gratuito de Render: lo que hay que saber
- Se **duerme tras 15 min sin tráfico** y el primer request tarda ≈ 1 min: abre la app 10 min antes de exponer.
- Opcional: un monitor gratuito (UptimeRobot, cron-job.org) que pida `/health` cada 5-10 min la mantiene
  despierta (un servicio 24/7 consume ≈ 744 de las 750 h gratuitas del mes).
- 512 MB de RAM: la app usa ≈ 120 MB con un modelo cargado (medido en local; los modelos se cargan por ciudad bajo demanda).
- El disco es efímero: el estado vive en git (modelos y datos versionados); nada se escribe en el servidor.

## 8. Verificación final (checklist)
- [ ] CI verde en `main` (lint, pruebas, build Docker).
- [ ] CD verde y el servicio en estado *Live* en Render.
- [ ] `/cities` muestra `available: true` en las ciudades entrenadas.
- [ ] `/model-info?city=<ciudad>` muestra versión, métricas y rango de datos reales.
- [ ] `/latest?city=<ciudad>` devuelve datos recientes.
- [ ] `retrain.yml` ejecutado manualmente al menos una vez (Actions → Retrain → Run workflow).
