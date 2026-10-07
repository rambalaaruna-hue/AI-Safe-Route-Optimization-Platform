"""Project-wide paths and modeling constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "real_accident_data.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "accident_model.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Routing: higher values prefer safer (longer) detours over risky shortcuts.
RISK_PENALTY_ALPHA = 8.5
RISK_PENALTY_BETA = 1.15
AVERAGE_CITY_SPEED_KMPH = 28.0

WEATHER_VALUES = ("Clear", "Cloudy", "Rain", "Heavy Rain", "Fog", "Storm")
LIGHTING_VALUES = ("Daylight", "Dusk", "Night Lit", "Night Unlit")
ROAD_SURFACE_VALUES = ("Dry", "Damp", "Wet", "Flooded")
ROAD_TYPE_VALUES = ("Arterial", "Collector", "Local", "Highway")
JUNCTION_VALUES = ("None", "T-Junction", "Cross", "Roundabout", "Flyover")
URBAN_RURAL_VALUES = ("Urban", "Suburban", "Rural")
DAY_VALUES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

CATEGORICAL_FEATURES = [
    "weather",
    "lighting",
    "road_surface",
    "road_type",
    "junction_type",
    "urban_rural",
    "day_of_week",
]

NUMERIC_FEATURES = [
    "hour",
    "month",
    "is_weekend",
    "is_night",
    "is_peak_hour",
    "speed_limit",
    "traffic_density",
    "visibility_km",
    "precipitation_mm",
    "temperature_c",
    "humidity",
    "vehicle_count",
    "latitude",
    "longitude",
]

TARGET_COLUMN = "risk_score"

COLUMN_ALIASES = {
    "weather": ["weather", "weather_condition", "weather_conditions", "Weather"],
    "lighting": ["lighting", "light_conditions", "light_condition", "Lighting"],
    "road_surface": ["road_surface", "road_surface_conditions", "surface"],
    "road_type": ["road_type", "roadclass", "road_class"],
    "junction_type": ["junction_type", "junction_detail", "junction"],
    "urban_rural": ["urban_rural", "urban_or_rural_area", "area_type"],
    "day_of_week": ["day_of_week", "day", "weekday"],
    "hour": ["hour", "Hour"],
    "time": ["time", "Time", "accident_time"],
    "month": ["month", "Month"],
    "speed_limit": ["speed_limit", "speedlimit", "Speed_limit"],
    "traffic_density": ["traffic_density", "traffic", "congestion"],
    "visibility_km": ["visibility_km", "visibility"],
    "precipitation_mm": ["precipitation_mm", "rainfall", "precip"],
    "temperature_c": ["temperature_c", "temperature", "temp"],
    "humidity": ["humidity", "Humidity"],
    "vehicle_count": ["vehicle_count", "number_of_vehicles", "vehicles"],
    "latitude": ["latitude", "lat", "start_lat"],
    "longitude": ["longitude", "lon", "lng", "start_lng"],
    "risk_score": ["risk_score", "accident_risk", "risk"],
    "severity": ["severity", "accident_severity", "Severity"],
}
