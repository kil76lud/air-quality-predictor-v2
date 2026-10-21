"""Entrypoint de producción: API FastAPI + interfaz Gradio montada en "/".

Las rutas de la API (/health, /model-info, /predict, /latest, /docs) se registran primero y
tienen prioridad; la UI de Gradio (app/ui.py -> build_ui()) atiende el resto.
Si la UI falla al cargar, la API sigue funcionando (se registra el error).
"""

from __future__ import annotations

import logging

from aqi.api import app

log = logging.getLogger("main")


def _mount_ui() -> None:
    try:
        import gradio as gr

        from app.ui import build_ui

        gr.mount_gradio_app(app, build_ui(), path="/")
    except Exception:  # noqa: BLE001 - la API debe sobrevivir a un fallo de la UI
        log.exception("UI could not be mounted; serving the API only.")


_mount_ui()
