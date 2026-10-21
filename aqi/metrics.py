"""Métricas de evaluación (accuracy, F1 macro, F1 por clase, matriz de confusión)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .constants import CLASSES


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Métricas sobre las 5 clases.

    El F1 macro se promedia SOLO sobre las clases presentes en y_true: en una ciudad de baja
    contaminación (p. ej. Puno) clases como "Very Unhealthy" pueden no aparecer en el test y
    un F1 = 0 artificial penalizaría injustamente a todos los modelos por igual.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    present = [c for c in CLASSES if c in set(yt)]
    f1_macro = f1_score(yt, yp, labels=present, average="macro", zero_division=0)
    per_class = f1_score(yt, yp, labels=CLASSES, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_macro),
        "f1_per_class": {c: float(v) for c, v in zip(CLASSES, per_class, strict=True)},
        "support": {c: int((yt == c).sum()) for c in CLASSES},
        "confusion_matrix": confusion_matrix(yt, yp, labels=CLASSES).tolist(),
    }
