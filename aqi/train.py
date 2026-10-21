"""Entrenamiento reproducible: baseline + XGBoost/Optuna (challenger) + champion/challenger.

Flujo de `train_pipeline`:
  1. Dataset -> features/target (aqi.features) -> split cronológico (con hueco = horizonte).
  2. Baseline: Random Forest sin ajustar (class_weight="balanced").
  3. Challenger: XGBoost con hiperparámetros de Optuna (objetivo = F1 macro medio en
     TimeSeriesSplit sobre el tramo de entrenamiento) y pesos de clase balanceados.
  4. Champion/challenger: ambos se evalúan en el MISMO test; se promueve solo si
     F1_challenger - F1_champion >= promotion_min_delta (0.005 por defecto).
  5. Todo queda en MLflow (params, métricas, artefactos, Registry con alias "champion").
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from . import reporting
from .champion import (
    REGISTERED_MODEL_NAME,
    append_decision,
    decide,
    load_champion,
    promote,
)
from .config import Settings, get_settings, resolve_settings, to_iso
from .constants import CLASSES
from .features import build_features
from .ingest import load_dataset
from .metrics import evaluate
from .model import AqiClassifier, fit_classifier

log = logging.getLogger(__name__)
EXPERIMENT_NAME = "air-quality-aqi"


def _model_name(s: Settings) -> str:
    """Nombre en el Model Registry: un modelo registrado por ciudad (alias `champion` en cada uno)."""
    return f"{REGISTERED_MODEL_NAME}-{s.city.slug}" if s.paths.slug else REGISTERED_MODEL_NAME


# ----------------------------------------------------------------------------- splits / CV
def split_time_ordered(
    X: pd.DataFrame, y: pd.Series, test_fraction: float, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split cronológico. Las filas de train cumplen t + horizonte < inicio del test, de modo
    que ninguna etiqueta de train (que mira t+h) se solapa con el periodo de test."""
    n_test = max(1, int(len(X) * test_fraction))
    test_start = X.index[len(X) - n_test]
    train_mask = X.index < test_start - pd.Timedelta(hours=horizon)
    test_mask = X.index >= test_start
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def make_xgb(params: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=2,
        random_state=seed,
        verbosity=0,
    )


def cv_score(params: dict[str, Any], X: pd.DataFrame, y: pd.Series, n_splits: int,
             horizon: int, seed: int) -> float:
    """F1 macro medio en TimeSeriesSplit (con gap = horizonte). Folds con <2 clases se omiten."""
    scores: list[float] = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(X):
        if y.iloc[tr].nunique() < 2:
            continue
        model = fit_classifier(make_xgb(params, seed), X.iloc[tr], y.iloc[tr], True)
        scores.append(evaluate(y.iloc[va], model.predict(X.iloc[va]))["f1_macro"])
    return float(np.mean(scores)) if scores else 0.0


def run_optuna(X: pd.DataFrame, y: pd.Series, n_trials: int, n_splits: int, horizon: int,
               seed: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
        return cv_score(params, X, y, n_splits, horizon, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


# ----------------------------------------------------------------------------- MLflow
def _setup_mlflow(s: Settings):
    """Configura MLflow (SQLite + artefactos locales). Normaliza la ruta de artefactos del
    experimento si el repo se movió de máquina (laptop <-> runner de CI)."""
    import mlflow

    db = s.paths.mlflow_db
    desired = s.paths.mlflow_artifacts.resolve().as_uri()
    if db.exists():
        con = sqlite3.connect(db)
        try:
            con.execute("UPDATE experiments SET artifact_location=? WHERE name=?",
                        (desired, EXPERIMENT_NAME))
            con.commit()
        except sqlite3.OperationalError:  # esquema aún no migrado
            pass
        finally:
            con.close()
    mlflow.set_tracking_uri(f"sqlite:///{db.resolve()}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=desired)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def _flat_metrics(prefix: str, m: dict[str, Any]) -> dict[str, float]:
    out = {f"{prefix}accuracy": m["accuracy"], f"{prefix}f1_macro": m["f1_macro"]}
    for cls, v in m["f1_per_class"].items():
        out[f"{prefix}f1_{cls.lower().replace(' ', '_')}"] = v
    return out


def _log_mlflow(s: Settings, *, run_tag: str, info: dict[str, Any], challenger: AqiClassifier,
                study: optuna.Study, figures: dict[str, Path]) -> tuple[str, str]:
    """Registra baseline y challenger. Devuelve (run_id_challenger, version_en_registry)."""
    mlflow = _setup_mlflow(s)
    common = {"city": s.city.name, "horizon_hours": s.horizon_hours, "n_rows": info["n_rows"],
              "n_train": info["n_train"], "n_test": info["n_test"], "seed": info["seed"],
              "data_source": info["data_source"]}
    with mlflow.start_run(run_name=f"baseline-rf-{s.city.slug}-{run_tag}"):
        mlflow.set_tag("role", "baseline")
        mlflow.log_params({**common, "model": "RandomForest(default, class_weight=balanced)"})
        mlflow.log_metrics(_flat_metrics("test_", info["baseline"]))
        for key in ("class_distribution", "confusion_baseline"):
            mlflow.log_artifact(str(figures[key]))
    with mlflow.start_run(run_name=f"challenger-xgb-{s.city.slug}-{run_tag}") as run:
        mlflow.set_tag("role", "challenger")
        mlflow.log_params({**common, "model": "XGBoost+Optuna", "n_trials": info["n_trials"],
                           "cv_splits": s.training.n_splits,
                           **{f"xgb_{k}": v for k, v in study.best_params.items()}})
        for i, t in enumerate(study.trials):
            mlflow.log_metric("optuna_trial_cv_f1_macro", float(t.value or 0.0), step=i)
        mlflow.log_metric("cv_f1_macro_best", study.best_value)
        mlflow.log_metrics(_flat_metrics("test_", info["challenger"]))
        for key in ("confusion_challenger", "baseline_vs_challenger", "optuna_history",
                    "feature_importance"):
            mlflow.log_artifact(str(figures[key]))
        mlflow.sklearn.log_model(challenger, artifact_path="model",
                                 registered_model_name=_model_name(s))
        run_id = run.info.run_id
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{_model_name(s)}'")
    version = str(max(int(v.version) for v in versions if v.run_id == run_id))
    return run_id, version


def _mlflow_set_decision(s: Settings, version: str, promoted: bool, reason: str) -> None:
    mlflow = _setup_mlflow(s)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(_model_name(s), version, "decision",
                                 "promoted" if promoted else "rejected")
    client.set_model_version_tag(_model_name(s), version, "reason", reason[:250])
    if promoted:
        client.set_registered_model_alias(_model_name(s), "champion", version)


# ----------------------------------------------------------------------------- pipeline
def train_pipeline(
    *,
    dataset: pd.DataFrame | None = None,
    settings: Settings | None = None,
    n_trials: int | None = None,
    seed: int | None = None,
    min_delta: float | None = None,
    force_promote: bool = False,
    use_mlflow: bool = True,
    champion_dir: Path | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    cfg = s.training
    n_trials = cfg.n_trials if n_trials is None else n_trials
    seed = cfg.seed if seed is None else seed
    min_delta = cfg.promotion_min_delta if min_delta is None else min_delta
    champ_dir = champion_dir or s.paths.champion_dir
    horizon = s.horizon_hours

    df = dataset if dataset is not None else load_dataset(s)
    X, y = build_features(df, horizon)
    X_tr, y_tr, X_te, y_te = split_time_ordered(X, y, cfg.test_fraction, horizon)
    if len(X_tr) < 200 or len(X_te) < 50:
        raise ValueError(f"Not enough data to train (train={len(X_tr)}, test={len(X_te)}).")
    log.info("Train rows=%s test rows=%s | classes in test: %s", len(X_tr), len(X_te),
             y_te.value_counts().to_dict())

    # 1) Baseline: Random Forest sin ajustar
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1,
                                random_state=seed)
    baseline = fit_classifier(rf, X_tr, y_tr, balanced_weights=False)
    base_m = evaluate(y_te, baseline.predict(X_te))

    # 2) Challenger: XGBoost + Optuna
    study = run_optuna(X_tr, y_tr, n_trials, cfg.n_splits, horizon, seed)
    challenger = fit_classifier(make_xgb(study.best_params, seed), X_tr, y_tr, True)
    chall_m = evaluate(y_te, challenger.predict(X_te))

    # 3) Champion actual sobre el MISMO test
    incumbent = load_champion(champ_dir)
    champ_f1: float | None = None
    champ_m: dict[str, Any] | None = None
    if incumbent is not None:
        try:
            champ_m = evaluate(y_te, incumbent[0].predict(X_te))
            champ_f1 = champ_m["f1_macro"]
        except KeyError:  # el champion usa un set de features distinto -> incompatible
            log.warning("Incumbent champion has incompatible features; it will be replaced.")
    promoted, reason = decide(champ_f1, chall_m["f1_macro"], min_delta)
    if force_promote and not promoted:
        promoted, reason = True, f"Forced promotion (--force-promote). Original verdict: {reason}"

    # 4) Figuras
    rep = s.paths.reports_dir
    figures = {
        "class_distribution": reporting.plot_class_distribution(y, y_te, rep / "class_distribution.png"),
        "confusion_baseline": reporting.plot_confusion(base_m["confusion_matrix"], "Baseline (RF)", rep / "confusion_baseline.png"),
        "confusion_challenger": reporting.plot_confusion(chall_m["confusion_matrix"], "Challenger (XGBoost+Optuna)", rep / "confusion_challenger.png"),
        "baseline_vs_challenger": reporting.plot_baseline_vs_challenger(base_m, chall_m, rep / "baseline_vs_challenger.png"),
        "optuna_history": reporting.plot_optuna_history([t.value or 0.0 for t in study.trials], rep / "optuna_history.png"),
        "feature_importance": reporting.plot_importances(challenger.feature_names, challenger.feature_importances_, rep / "feature_importance.png"),
    }

    # 5) MLflow + versión
    run_tag = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    data_source = _data_source(s)
    info = {"n_rows": len(df), "n_train": len(X_tr), "n_test": len(X_te), "seed": seed,
            "n_trials": n_trials, "baseline": base_m, "challenger": chall_m,
            "data_source": data_source}
    prev_version = int(incumbent[1]["version"]) if incumbent else 0
    run_id, version = "", str(prev_version + 1)
    if use_mlflow:
        run_id, version = _log_mlflow(s, run_tag=run_tag, info=info, challenger=challenger,
                                      study=study, figures=figures)
        _mlflow_set_decision(s, version, promoted, reason)

    # 6) Promoción (escribe models/champion) + bitácora
    if promoted:
        metadata = {
            "model_name": "aqi-xgboost-champion",
            "version": version,
            "alias": "champion",
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": {"accuracy": chall_m["accuracy"], "f1_macro": chall_m["f1_macro"],
                        "f1_per_class": chall_m["f1_per_class"], "support": chall_m["support"]},
            "horizon_hours": horizon,
            "classes": CLASSES,
            "data_range": {"start": to_iso(df["time"].min(), s), "end": to_iso(df["time"].max(), s)},
            "test_range": {"start": to_iso(X_te.index.min(), s), "end": to_iso(X_te.index.max(), s)},
            "n_rows": int(len(df)),
            "feature_names": challenger.feature_names,
            "classes_trained": challenger.classes,
            "params": study.best_params,
            "seed": seed,
            "mlflow_run_id": run_id,
            "data_source": data_source,
            "city": s.city.name,
        }
        promote(challenger, metadata, champ_dir, df)

    decision = {
        "city": s.city.name,
        "promoted": promoted,
        "reason": reason,
        "version": version,
        "challenger_f1_macro": chall_m["f1_macro"],
        "champion_f1_macro": champ_f1,
        "baseline_f1_macro": base_m["f1_macro"],
        "min_delta": min_delta,
        "n_trials": n_trials,
        "seed": seed,
        "data_source": data_source,
        "mlflow_run_id": run_id,
    }
    append_decision(decision, rep)
    (rep / "last_run.json").write_text(json.dumps({**decision, "baseline": base_m,
                                                   "challenger": chall_m, "champion": champ_m},
                                                  indent=2), encoding="utf-8")
    return decision


def _data_source(s: Settings) -> str:
    if s.paths.raw_meta.exists():
        return json.loads(s.paths.raw_meta.read_text(encoding="utf-8")).get("source", "unknown")
    return "unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train challenger and run champion/challenger gate.")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-delta", type=float, default=None,
                   help="Minimum F1-macro improvement to promote (default from config).")
    p.add_argument("--force-promote", action="store_true", help="Promote regardless (first run).")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--city", default=None, help="City slug from config.yaml (default: the default city).")
    p.add_argument("--all", action="store_true", help="Train every city in config.yaml.")
    args = p.parse_args()
    decisions: list[dict[str, Any]] = []
    failed: list[str] = []
    for s in resolve_settings(args.city, args.all):
        print(f"=== {s.city.name} ===")
        try:
            d = train_pipeline(settings=s, n_trials=args.n_trials, seed=args.seed, min_delta=args.min_delta,
                               force_promote=args.force_promote, use_mlflow=not args.no_mlflow)
        except Exception as exc:  # noqa: BLE001 - una ciudad que falla no detiene a las demás
            log.error("Training failed for %s: %s", s.city.name, exc)
            failed.append(s.city.name)
            continue
        decisions.append(d)
        print(json.dumps(d, indent=2))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:  # expone el resultado a GitHub Actions: promoted = alguna ciudad tuvo nuevo champion
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"promoted={'true' if any(d['promoted'] for d in decisions) else 'false'}\n")
    if failed:
        print(f"TRAINING FAILED for: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
