"""Shared feature cleaning and engineering used by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    COLUMN_ALIASES,
    DAY_VALUES,
    JUNCTION_VALUES,
    LIGHTING_VALUES,
    NUMERIC_FEATURES,
    ROAD_SURFACE_VALUES,
    ROAD_TYPE_VALUES,
    TARGET_COLUMN,
    URBAN_RURAL_VALUES,
    WEATHER_VALUES,
)

_ALLOWED = {
    "weather": set(WEATHER_VALUES),
    "lighting": set(LIGHTING_VALUES),
    "road_surface": set(ROAD_SURFACE_VALUES),
    "road_type": set(ROAD_TYPE_VALUES),
    "junction_type": set(JUNCTION_VALUES),
    "urban_rural": set(URBAN_RURAL_VALUES),
    "day_of_week": set(DAY_VALUES),
}

_WEATHER_MAP = {
    "fine": "Clear",
    "clear": "Clear",
    "sunny": "Clear",
    "overcast": "Cloudy",
    "cloudy": "Cloudy",
    "rain": "Rain",
    "raining": "Rain",
    "light rain": "Rain",
    "heavy rain": "Heavy Rain",
    "fog": "Fog",
    "foggy": "Fog",
    "mist": "Fog",
    "storm": "Storm",
    "thunder": "Storm",
    "snow": "Storm",
}

_LIGHT_MAP = {
    "daylight": "Daylight",
    "day": "Daylight",
    "dusk": "Dusk",
    "dawn": "Dusk",
    "darkness - lights lit": "Night Lit",
    "night lit": "Night Lit",
    "street lights": "Night Lit",
    "darkness - no lighting": "Night Unlit",
    "night unlit": "Night Unlit",
    "dark": "Night Unlit",
}


def _first_matching_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _normalize_weather(value: object) -> str:
    text = str(value).strip()
    mapped = _WEATHER_MAP.get(text.lower())
    if mapped:
        return mapped
    return text if text in _ALLOWED["weather"] else "Clear"


def _normalize_lighting(value: object) -> str:
    text = str(value).strip()
    mapped = _LIGHT_MAP.get(text.lower())
    if mapped:
        return mapped
    return text if text in _ALLOWED["lighting"] else "Daylight"


def _normalize_category(value: object, field: str, default: str) -> str:
    text = str(value).strip()
    if text in _ALLOWED[field]:
        return text
    lowered = text.lower()
    for option in _ALLOWED[field]:
        if option.lower() == lowered:
            return option
    return default


def _hour_from_time(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        hour = float(value)
        return hour if 0 <= hour <= 23 else np.nan
    text = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p"):
        try:
            return float(pd.to_datetime(text, format=fmt).hour)
        except (ValueError, TypeError):
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour)


def _severity_to_risk(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        min_v, max_v = numeric.min(), numeric.max()
        if pd.isna(min_v) or max_v == min_v:
            return pd.Series(np.full(len(series), 0.35), index=series.index)
        scaled = (numeric - min_v) / (max_v - min_v)
        return scaled.clip(0, 1)

    mapping = {
        "1": 0.92,
        "2": 0.62,
        "3": 0.28,
        "fatal": 0.95,
        "serious": 0.72,
        "slight": 0.28,
        "severe": 0.78,
        "moderate": 0.48,
        "minor": 0.22,
        "low": 0.18,
        "medium": 0.48,
        "high": 0.82,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(0.40)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map heterogeneous accident CSVs onto the project feature schema."""
    resolved: dict[str, pd.Series] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        source = _first_matching_column(df, aliases)
        if source is not None:
            resolved[canonical] = df[source]

    out = pd.DataFrame(index=df.index)

    if "hour" in resolved:
        out["hour"] = pd.to_numeric(resolved["hour"], errors="coerce")
    elif "time" in resolved:
        out["hour"] = resolved["time"].map(_hour_from_time)
    else:
        out["hour"] = 12

    out["hour"] = out["hour"].fillna(12).clip(0, 23).astype(float)

    if "month" in resolved:
        out["month"] = pd.to_numeric(resolved["month"], errors="coerce").fillna(6).clip(1, 12)
    else:
        out["month"] = 6

    if "day_of_week" in resolved:
        out["day_of_week"] = resolved["day_of_week"].map(
            lambda v: _normalize_category(v, "day_of_week", "Wednesday")
        )
    else:
        out["day_of_week"] = "Wednesday"

    weekend_days = {"Saturday", "Sunday"}
    out["is_weekend"] = out["day_of_week"].isin(weekend_days).astype(int)
    out["is_night"] = ((out["hour"] >= 20) | (out["hour"] < 6)).astype(int)
    out["is_peak_hour"] = (
        ((out["hour"] >= 8) & (out["hour"] <= 10)) | ((out["hour"] >= 17) & (out["hour"] <= 20))
    ).astype(int)

    out["weather"] = (
        resolved["weather"].map(_normalize_weather) if "weather" in resolved else "Clear"
    )
    out["lighting"] = (
        resolved["lighting"].map(_normalize_lighting) if "lighting" in resolved else "Daylight"
    )
    out["road_surface"] = (
        resolved["road_surface"].map(lambda v: _normalize_category(v, "road_surface", "Dry"))
        if "road_surface" in resolved
        else "Dry"
    )
    out["road_type"] = (
        resolved["road_type"].map(lambda v: _normalize_category(v, "road_type", "Arterial"))
        if "road_type" in resolved
        else "Arterial"
    )
    out["junction_type"] = (
        resolved["junction_type"].map(lambda v: _normalize_category(v, "junction_type", "None"))
        if "junction_type" in resolved
        else "None"
    )
    out["urban_rural"] = (
        resolved["urban_rural"].map(lambda v: _normalize_category(v, "urban_rural", "Urban"))
        if "urban_rural" in resolved
        else "Urban"
    )

    defaults = {
        "speed_limit": 50,
        "traffic_density": 0.45,
        "visibility_km": 8.0,
        "precipitation_mm": 0.0,
        "temperature_c": 28.0,
        "humidity": 60.0,
        "vehicle_count": 2,
        "latitude": 17.44,
        "longitude": 78.45,
    }
    for name, default in defaults.items():
        if name in resolved:
            out[name] = pd.to_numeric(resolved[name], errors="coerce").fillna(default)
        else:
            out[name] = default

    out["speed_limit"] = out["speed_limit"].clip(20, 120)
    out["traffic_density"] = out["traffic_density"].clip(0, 1)
    out["visibility_km"] = out["visibility_km"].clip(0.05, 20)
    out["precipitation_mm"] = out["precipitation_mm"].clip(0, 120)
    out["humidity"] = out["humidity"].clip(10, 100)
    out["vehicle_count"] = out["vehicle_count"].clip(1, 12)

    if TARGET_COLUMN in resolved:
        out[TARGET_COLUMN] = pd.to_numeric(resolved[TARGET_COLUMN], errors="coerce")
        if out[TARGET_COLUMN].isna().all() and "severity" in resolved:
            out[TARGET_COLUMN] = _severity_to_risk(resolved["severity"])
        else:
            out[TARGET_COLUMN] = out[TARGET_COLUMN].fillna(out[TARGET_COLUMN].median())
            if out[TARGET_COLUMN].max() > 1.5:
                out[TARGET_COLUMN] = out[TARGET_COLUMN] / out[TARGET_COLUMN].max()
            out[TARGET_COLUMN] = out[TARGET_COLUMN].clip(0, 1)
    elif "severity" in resolved:
        out[TARGET_COLUMN] = _severity_to_risk(resolved["severity"])
    else:
        out[TARGET_COLUMN] = np.nan

    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    engineered = standardize_dataframe(df)
    return engineered[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def build_inference_row(
    *,
    weather: str,
    lighting: str,
    road_surface: str,
    road_type: str,
    junction_type: str,
    urban_rural: str,
    day_of_week: str,
    hour: int,
    month: int,
    speed_limit: float,
    traffic_density: float,
    visibility_km: float,
    precipitation_mm: float,
    temperature_c: float,
    humidity: float,
    vehicle_count: int,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    payload = {
        "weather": weather,
        "lighting": lighting,
        "road_surface": road_surface,
        "road_type": road_type,
        "junction_type": junction_type,
        "urban_rural": urban_rural,
        "day_of_week": day_of_week,
        "hour": hour,
        "month": month,
        "speed_limit": speed_limit,
        "traffic_density": traffic_density,
        "visibility_km": visibility_km,
        "precipitation_mm": precipitation_mm,
        "temperature_c": temperature_c,
        "humidity": humidity,
        "vehicle_count": vehicle_count,
        "latitude": latitude,
        "longitude": longitude,
        "risk_score": 0.0,
    }
    return feature_frame(pd.DataFrame([payload]))
