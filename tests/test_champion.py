import numpy as np
import pandas as pd
import pytest

from aqi.champion import decide, load_champion
from aqi.constants import CLASSES, FEATURE_COLUMNS
from aqi.model import fit_classifier
from aqi.train import train_pipeline


def test_first_model_is_promoted():
    ok, reason = decide(None, 0.5, 0.005)
    assert ok and "first model" in reason


@pytest.mark.parametrize(
    "champ,chall,delta,expected",
    [(0.600, 0.606, 0.005, True), (0.600, 0.605, 0.005, True), (0.600, 0.604, 0.005, False),
     (0.600, 0.600, 0.005, False), (0.600, 0.550, 0.005, False), (0.600, 0.550, -1.0, True)],
)
def test_promotion_threshold(champ, chall, delta, expected):
    ok, reason = decide(champ, chall, delta)
    assert ok is expected
    assert ("Promoted" in reason) is expected and ("Rejected" in reason) is not expected


def test_classifier_handles_missing_classes_and_returns_five_probabilities():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series(np.where(X.iloc[:, 0] > 0, "Good", "Unhealthy"))  # solo 2 de 5 clases
    model = fit_classifier(RandomForestClassifier(n_estimators=10, random_state=0), X, y, False)
    proba = model.predict_proba(X.head(5))
    assert proba.shape == (5, 5) and np.allclose(proba.sum(axis=1), 1)
    assert proba[:, CLASSES.index("Very Unhealthy")].sum() == 0


def test_same_data_and_seed_yield_rejected_challenger(synthetic_df, trained_champion, tmp_settings):
    """Reproducibilidad: reentrenar con los mismos datos/seed no mejora -> se descarta."""
    again = train_pipeline(dataset=synthetic_df, settings=tmp_settings, n_trials=3, use_mlflow=False)
    assert again["promoted"] is False and "Rejected" in again["reason"]


def test_decisions_are_logged(trained_champion, tmp_settings):
    log_file = tmp_settings.paths.reports_dir / "decisions.jsonl"
    assert log_file.exists() and log_file.read_text().strip()
    assert load_champion(trained_champion[0]) is not None
