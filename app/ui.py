"""Interfaz de Aire Puno (diseño final).

Contrato con main.py: este módulo expone `build_ui() -> gradio.Blocks`.
La página vive en app/web/index.html (HTML + CSS + JS, sin build) y consume la API por HTTP
(GET /latest, POST /predict, GET /model-info, GET /health) con rutas relativas, por lo que
funciona igual en local, Docker y Render.

Se incrusta en un <iframe srcdoc> para tener control total del diseño sin pelear con los
estilos de Gradio. Para cambiar el diseño, edita solo app/web/index.html.
"""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

WEB_FILE = Path(__file__).parent / "web" / "index.html"

CSS = """
footer, .built-with, .api-docs, .show-api {display: none !important;}
.gradio-container {max-width: 100% !important; padding: 0 !important; margin: 0 !important;}
.gradio-container main, .gradio-container .main, .gradio-container .wrap {padding: 0 !important; gap: 0 !important;}
#aqi-frame {padding: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important;}
#aqi-frame iframe {display: block; width: 100%; height: 100vh; border: 0;}
body {margin: 0 !important;}
"""


def build_ui() -> gr.Blocks:
    page = WEB_FILE.read_text(encoding="utf-8")
    frame = f'<iframe title="Aire Puno" srcdoc="{html.escape(page, quote=True)}"></iframe>'
    with gr.Blocks(title="Aire Puno · Air Quality Category Predictor", css=CSS, fill_width=True) as demo:
        gr.HTML(frame, elem_id="aqi-frame", padding=False)
    return demo
