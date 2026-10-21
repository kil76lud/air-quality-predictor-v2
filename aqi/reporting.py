"""Figuras para la demo y el informe (se guardan en models/reports y en MLflow)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (CI / Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .constants import CLASSES  # noqa: E402

SHORT = ["Good", "Moderate", "USG", "Unhealthy", "Very Unh."]


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_class_distribution(y_all: pd.Series, y_test: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CLASSES))
    all_c = [(y_all == c).sum() for c in CLASSES]
    test_c = [(y_test == c).sum() for c in CLASSES]
    ax.bar(x - 0.2, all_c, 0.4, label="All data")
    ax.bar(x + 0.2, test_c, 0.4, label="Test set")
    ax.set_xticks(x, SHORT)
    ax.set_ylabel("Hours")
    ax.set_title("Target class distribution (AQI category at t+6h)")
    ax.legend()
    return _save(fig, path)


def plot_confusion(cm: Sequence[Sequence[int]], title: str, path: Path) -> Path:
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), SHORT, rotation=30)
    ax.set_yticks(range(len(CLASSES)), SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    return _save(fig, path)


def plot_baseline_vs_challenger(baseline: dict, challenger: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = ["accuracy", "f1_macro"]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [baseline[n] for n in names], 0.4, label="Baseline (RF default)")
    ax.bar(x + 0.2, [challenger[n] for n in names], 0.4, label="Optimized (XGBoost + Optuna)")
    ax.set_xticks(x, ["Accuracy", "F1 macro"])
    ax.set_ylim(0, 1)
    ax.set_title("Baseline vs optimized model (test set)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_optuna_history(values: Sequence[float], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    v = np.asarray(values, dtype=float)
    ax.scatter(range(len(v)), v, s=18, label="Trial F1 macro (CV)")
    ax.plot(np.maximum.accumulate(v), color="C3", label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("F1 macro (TimeSeriesSplit)")
    ax.set_title("Optuna optimization history")
    ax.legend()
    return _save(fig, path)


def plot_importances(names: Sequence[str], importances: Sequence[float], path: Path, top: int = 20) -> Path:
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order], [importances[i] for i in order])
    ax.set_title(f"Top {top} feature importances (XGBoost)")
    return _save(fig, path)
