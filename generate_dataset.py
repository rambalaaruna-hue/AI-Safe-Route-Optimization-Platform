"""Synthesize a large, production-style accident dataset for model training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DATA_PATH,
    DAY_VALUES,
    JUNCTION_VALUES,
    LIGHTING_VALUES,
    RANDOM_STATE,
    ROAD_SURFACE_VALUES,
    ROAD_TYPE_VALUES,
    URBAN_RURAL_VALUES,
    WEATHER_VALUES,
)

N_ROWS = 80_000
HYDERABAD_LAT_RANGE = (17.30, 17.54)
HYDERABAD_LON_RANGE = (78.32, 78.58)


def _risk_from_features(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    weather_w = df["weather"].map(
        {"Clear": 0.00, "Cloudy": 0.04, "Rain": 0.13, "Heavy Rain": 0.22, "Fog": 0.20, "Storm": 0.28}
    ).to_numpy(dtype=float)
    light_w = df["lighting"].map(
        {"Daylight": 0.00, "Dusk": 0.05, "Night Lit": 0.09, "Night Unlit": 0.18}
    ).to_numpy(dtype=float)
    surface_w = df["road_surface"].map(
        {"Dry": 0.00, "Damp": 0.05, "Wet": 0.12, "Flooded": 0.24}
    ).to_numpy(dtype=float)
    road_w = df["road_type"].map(
        {"Local": 0.03, "Collector": 0.05, "Arterial": 0.08, "Highway": 0.12}
    ).to_numpy(dtype=float)
    junction_w = df["junction_type"].map(
        {"None": 0.02, "T-Junction": 0.08, "Cross": 0.12, "Roundabout": 0.06, "Flyover": 0.04}
    ).to_numpy(dtype=float)
    area_w = df["urban_rural"].map({"Urban": 0.07, "Suburban": 0.04, "Rural": 0.09}).to_numpy(dtype=float)

    hour = df["hour"].to_numpy(dtype=float)
    night = ((hour >= 20) | (hour < 6)).astype(float)
    peak = (((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))).astype(float)
    weekend = df["day_of_week"].isin(["Saturday", "Sunday"]).to_numpy(dtype=float)

    traffic = df["traffic_density"].to_numpy(dtype=float)
    visibility = df["visibility_km"].to_numpy(dtype=float)
    precip = df["precipitation_mm"].to_numpy(dtype=float)
    speed = df["speed_limit"].to_numpy(dtype=float)

    score = (
        0.07
        + weather_w
        + light_w
        + surface_w
        + road_w
        + junction_w
        + area_w
        + 0.10 * night
        + 0.07 * peak
        + 0.04 * weekend * night
        + 0.18 * traffic
        + 0.12 * np.clip((10.0 - visibility) / 10.0, 0, 1)
        + 0.08 * np.clip(precip / 40.0, 0, 1)
        + 0.10 * np.clip((speed - 40.0) / 60.0, 0, 1)
        + 0.06 * traffic * weather_w
    )
    noise = rng.normal(0.0, 0.035, size=len(df))
    return np.clip(score + noise, 0.01, 0.99)


def generate_dataset(n_rows: int = N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    weather = rng.choice(WEATHER_VALUES, size=n_rows, p=[0.42, 0.18, 0.18, 0.08, 0.08, 0.06])
    lighting = rng.choice(LIGHTING_VALUES, size=n_rows, p=[0.52, 0.12, 0.24, 0.12])
    road_surface = rng.choice(ROAD_SURFACE_VALUES, size=n_rows, p=[0.55, 0.15, 0.22, 0.08])
    road_type = rng.choice(ROAD_TYPE_VALUES, size=n_rows, p=[0.34, 0.28, 0.22, 0.16])
    junction_type = rng.choice(JUNCTION_VALUES, size=n_rows, p=[0.30, 0.22, 0.22, 0.16, 0.10])
    urban_rural = rng.choice(URBAN_RURAL_VALUES, size=n_rows, p=[0.62, 0.28, 0.10])
    day_of_week = rng.choice(DAY_VALUES, size=n_rows)

    hour = rng.integers(0, 24, size=n_rows)
    month = rng.integers(1, 13, size=n_rows)
    speed_limit = rng.choice([30, 40, 50, 60, 80, 100], size=n_rows, p=[0.12, 0.22, 0.28, 0.20, 0.12, 0.06])
    traffic_density = np.clip(rng.beta(2.2, 2.6, size=n_rows), 0, 1)
    visibility_km = np.clip(rng.normal(8.5, 3.2, size=n_rows), 0.2, 18.0)
    precipitation_mm = np.clip(rng.gamma(1.4, 4.5, size=n_rows) * (weather != "Clear"), 0, 90)
    temperature_c = np.clip(rng.normal(29.0, 4.5, size=n_rows), 14, 44)
    humidity = np.clip(rng.normal(62, 14, size=n_rows), 18, 98)
    vehicle_count = rng.integers(1, 7, size=n_rows)
    latitude = rng.uniform(*HYDERABAD_LAT_RANGE, size=n_rows)
    longitude = rng.uniform(*HYDERABAD_LON_RANGE, size=n_rows)

    fog_or_storm = np.isin(weather, ["Fog", "Storm", "Heavy Rain"])
    visibility_km = np.where(fog_or_storm, np.minimum(visibility_km, rng.uniform(0.3, 3.5, n_rows)), visibility_km)
    wet_mask = np.isin(weather, ["Rain", "Heavy Rain", "Storm"])
    road_surface = np.where(wet_mask & (rng.random(n_rows) < 0.75), "Wet", road_surface)

    df = pd.DataFrame(
        {
            "accident_id": np.arange(1, n_rows + 1),
            "latitude": latitude.round(6),
            "longitude": longitude.round(6),
            "hour": hour,
            "month": month,
            "day_of_week": day_of_week,
            "weather": weather,
            "lighting": lighting,
            "road_surface": road_surface,
            "road_type": road_type,
            "junction_type": junction_type,
            "urban_rural": urban_rural,
            "speed_limit": speed_limit,
            "traffic_density": traffic_density.round(3),
            "visibility_km": visibility_km.round(2),
            "precipitation_mm": precipitation_mm.round(2),
            "temperature_c": temperature_c.round(1),
            "humidity": humidity.round(1),
            "vehicle_count": vehicle_count,
        }
    )
    df["risk_score"] = _risk_from_features(df, rng).round(4)
    return df


def main() -> None:
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {DATA_PATH}")
    print(df["risk_score"].describe().to_string())


if __name__ == "__main__":
    main()
