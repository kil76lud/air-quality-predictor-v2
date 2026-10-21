"""Generador de datos SINTÉTICOS (solo tests y modelo placeholder inicial).

No sustituye a los datos reales: `make first-run` descarga el histórico real de Open-Meteo y
reemplaza el modelo placeholder. El us_aqi sintético se deriva de PM2.5 con los puntos de
corte de la EPA para que el problema sea aprendible.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import get_settings
from .download import write_bootstrap_meta
from .validation import validate_frame

# (c_lo, c_hi, aqi_lo, aqi_hi) puntos de corte EPA para PM2.5 (µg/m³)
_PM25_BREAKS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]


def pm25_to_aqi(pm: float) -> float:
    pm = round(pm, 1)
    for c_lo, c_hi, a_lo, a_hi in _PM25_BREAKS:
        if pm <= c_hi:
            return round((a_hi - a_lo) / (c_hi - c_lo) * (max(pm, c_lo) - c_lo) + a_lo)
    return 500.0


def make_synthetic(n_hours: int = 24 * 365, seed: int = 0, end: str = "2026-10-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(end=pd.Timestamp(end), periods=n_hours, freq="h")
    hour = t.hour.to_numpy()
    doy = t.dayofyear.to_numpy()

    # PM2.5: log-AR(1) persistente + ciclo diario/estacional + episodios de humo
    x = np.zeros(n_hours)
    for i in range(1, n_hours):
        x[i] = 0.985 * x[i - 1] + rng.normal(0, 0.12)
    episodes = np.zeros(n_hours)
    for start in rng.integers(0, n_hours, size=max(4, n_hours // 900)):
        length = int(rng.integers(12, 60))
        episodes[start : start + length] += rng.uniform(1.0, 2.6)
    log_pm = (
        2.2
        + 0.25 * np.sin(2 * np.pi * (hour - 6) / 24)
        + 0.35 * np.sin(2 * np.pi * (doy - 200) / 365)
        + x
        + episodes
    )
    pm25 = np.exp(log_pm)
    df = pd.DataFrame({"time": t})
    df["pm2_5"] = np.clip(pm25, 0, 600)
    df["pm10"] = np.clip(df["pm2_5"] * 1.8 + rng.normal(0, 2, n_hours), 0, 1500)
    df["co"] = np.clip(150 + 9 * df["pm2_5"] + rng.normal(0, 15, n_hours), 0, 20000)
    df["no2"] = np.clip(6 + 0.15 * df["pm2_5"] + rng.normal(0, 1.5, n_hours), 0, 500)
    df["so2"] = np.clip(1.5 + 0.03 * df["pm2_5"] + rng.normal(0, 0.4, n_hours), 0, 500)
    df["o3"] = np.clip(55 + 18 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 4, n_hours), 0, 400)
    df["temperature"] = 7 + 7 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.8, n_hours)
    df["humidity"] = np.clip(55 - 20 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 5, n_hours), 5, 100)
    df["wind_speed"] = np.clip(9 + 5 * np.sin(2 * np.pi * (hour - 12) / 24) + rng.normal(0, 2, n_hours), 0, 120)
    df["pressure"] = np.clip(648 + 2 * np.sin(2 * np.pi * hour / 24) + np.cumsum(rng.normal(0, 0.05, n_hours)) % 6, 300, 1100)
    df["us_aqi"] = [pm25_to_aqi(v) for v in df["pm2_5"]]
    return validate_frame(df)


def main() -> None:
    p = argparse.ArgumentParser(description="Write a SYNTHETIC bootstrap dataset (placeholder).")
    p.add_argument("--hours", type=int, default=24 * 365)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    s = get_settings()
    df = make_synthetic(args.hours, args.seed)
    s.paths.raw_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(s.paths.raw_file, index=False)
    write_bootstrap_meta(s, df, source="synthetic")
    print(f"Synthetic bootstrap written: {len(df)} rows")


if __name__ == "__main__":
    main()
