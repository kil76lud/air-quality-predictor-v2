"""Wrapper de clasificador robusto a clases ausentes.

XGBoost exige etiquetas 0..k-1 contiguas y presentes en el entrenamiento. En series temporales
algunas clases raras pueden faltar en un fold o en todo el histórico. Este wrapper codifica
internamente solo las clases presentes y SIEMPRE devuelve probabilidades para las 5 clases
(0.0 para las ausentes), cumpliendo el contrato de la API.

NOTA: este módulo debe conservar su ruta (aqi.model.AqiClassifier): joblib la necesita al cargar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from .constants import CLASSES


class AqiClassifier:
    def __init__(self, estimator: Any, feature_names: list[str], classes: list[str]) -> None:
        self.estimator = estimator
        self.feature_names = feature_names
        self.classes = classes  # clases vistas en entrenamiento, en el orden interno 0..k-1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(X[self.feature_names])
        out = np.zeros((len(X), len(CLASSES)))
        for j, cls in enumerate(self.classes):
            out[:, CLASSES.index(cls)] = raw[:, j]
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray(CLASSES, dtype=object)[proba.argmax(axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self.estimator.feature_importances_)


def fit_classifier(
    estimator: Any, X: pd.DataFrame, y: pd.Series, balanced_weights: bool
) -> AqiClassifier:
    """Ajusta `estimator` codificando solo las clases presentes.

    balanced_weights=True aplica pesos de clase "balanced" vía sample_weight (XGBoost no tiene
    class_weight nativo); el Random Forest usa class_weight="balanced" en su constructor.
    """
    classes = [c for c in CLASSES if c in set(y)]
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes to train, found {classes}")
    mapping = {c: i for i, c in enumerate(classes)}
    y_idx = y.map(mapping).to_numpy()
    if balanced_weights:
        estimator.fit(X, y_idx, sample_weight=compute_sample_weight("balanced", y_idx))
    else:
        estimator.fit(X, y_idx)
    return AqiClassifier(estimator, list(X.columns), classes)
