# Plan de trabajo — 2 días, 4 personas

**Roles:** **A** Datos/ML (aqi/download, ingest, features, train) · **B** CI/CD y despliegue
(workflows, Docker, HF, secrets) · **C** API + integración de la UI + pruebas · **D** Documentación,
informe, guion y ensayos. El repo ya incluye código, pruebas y workflows: el trabajo es
**ejecutar, validar con datos reales, desplegar, integrar la UI y ensayar**.

## Día 1 — Que todo funcione en producción
| Hora | A | B | C | D |
|---|---|---|---|---|
| 09:00-10:00 | Todos: instalar entorno, `make test`, leer README y contrato de API | | | |
| 10:00-12:00 | `make first-run` con datos reales; revisar clases y métricas; ajustar `n_trials` si hace falta | Crear repo GitHub, servicio Render (Blueprint), Deploy Hook y secrets; permisos de Actions | Revisar `/docs`; probar `/predict` y `/latest` en local; coordinar con el diseñador de UI | Esqueleto del informe (docs/REPORT_OUTLINE.md) |
| 12:00-13:00 | Commit de `data/`, `models/`, `mlflow.db` | Primer `git push` → CI verde | Verificar contrato con la UI | Capturas de MLflow y figuras |
| 14:00-16:00 | Pruebas extra si hay fallos con datos reales (clases ausentes, rangos) | **Primer despliegue (cd.yml)**; resolver errores de build | Recibir `app/ui.py`; probar con `make serve` | Redactar cap. 1 y 2 |
| 16:00-18:00 | Ejecutar `retrain.yml` manual (rechazado y promovido) | Verificar redeploy y `/model-info` | Integrar UI y pasar CI/CD | Cap. 3 y 4; capturas de Actions |
| 18:00 | **Hito:** app real desplegada con UI + un retrain exitoso | | | |

## Día 2 — Validar, documentar, ensayar
| Hora | A | B | C | D |
|---|---|---|---|---|
| 09:00-11:00 | Preparar figuras para la exposición (clases, baseline vs optimizado, Optuna, importancias) | Ensayar CI rojo/verde (quality gate) y runbook de despliegue | Pruebas de humo de la UI en inglés; casos de error | Completar informe + anexos |
| 11:00-13:00 | Responder preguntas técnicas (leakage, limitaciones) | Preparar plan B (capturas de cada paso) | Revisión de código y limpieza (ruff) | Diapositivas (≈ 10) |
| 14:00-15:30 | **Ensayo 1 cronometrado** (todos) con `docs/PRESENTATION_SCRIPT.md` | | | |
| 15:30-17:00 | Correcciones del ensayo | Congelar `main` (sin cambios riesgosos) | Verificar servicio Render *Live* | Entregar PDF del informe |
| 17:00-18:00 | **Ensayo 2** completo + preguntas cruzadas | | | |

## Reglas
- Congelar cambios en `main` a las 17:00 del día 2; solo hotfix con CI verde.
- Cualquier cambio al contrato de la API se avisa a C y al diseñador de UI antes de tocar código.
- Riesgos: Open-Meteo caído (usar `data/raw` versionado), servicio de Render dormido (abrirlo 10 min antes; arranque en frío ≈ 1 min), GitHub Actions lento (tener capturas).
