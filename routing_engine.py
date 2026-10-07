"""City-scale intersection graph and risk-aware A* / Dijkstra routing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd

from config import (
    AVERAGE_CITY_SPEED_KMPH,
    MODEL_PATH,
    RISK_PENALTY_ALPHA,
    RISK_PENALTY_BETA,
)
from feature_engineering import build_inference_row

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class RouteResult:
    node_ids: list[str]
    coordinates: list[tuple[float, float]]
    edge_records: list[dict[str, Any]]
    distance_km: float
    travel_time_min: float
    mean_risk: float
    max_risk: float
    cumulative_risk: float
    safety_index: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def lighting_from_hour(hour: int) -> str:
    if 6 <= hour < 8 or 17 <= hour < 19:
        return "Dusk"
    if 8 <= hour < 17:
        return "Daylight"
    return "Night Lit"


def surface_from_weather(weather: str) -> str:
    mapping = {
        "Clear": "Dry",
        "Cloudy": "Damp",
        "Rain": "Wet",
        "Heavy Rain": "Wet",
        "Fog": "Damp",
        "Storm": "Flooded",
    }
    return mapping.get(weather, "Dry")


def precipitation_from_weather(weather: str) -> float:
    return {"Clear": 0.0, "Cloudy": 0.4, "Rain": 8.5, "Heavy Rain": 24.0, "Fog": 1.2, "Storm": 38.0}.get(
        weather, 0.0
    )


def visibility_from_weather(weather: str) -> float:
    return {"Clear": 12.0, "Cloudy": 8.5, "Rain": 4.5, "Heavy Rain": 2.2, "Fog": 0.8, "Storm": 1.5}.get(
        weather, 8.0
    )


def build_city_graph() -> nx.Graph:
    """Hyderabad arterial network: 22 named intersections with real coordinates."""
    nodes: dict[str, dict[str, Any]] = {
        "Miyapur": {"lat": 17.4969, "lon": 78.3736, "area": "Suburban"},
        "Kukatpally": {"lat": 17.4948, "lon": 78.3996, "area": "Urban"},
        "Moosapet": {"lat": 17.4675, "lon": 78.4208, "area": "Urban"},
        "SR Nagar": {"lat": 17.4436, "lon": 78.4464, "area": "Urban"},
        "Ameerpet": {"lat": 17.4375, "lon": 78.4483, "area": "Urban"},
        "Panjagutta": {"lat": 17.4268, "lon": 78.4515, "area": "Urban"},
        "Begumpet": {"lat": 17.4440, "lon": 78.4672, "area": "Urban"},
        "Secunderabad": {"lat": 17.4399, "lon": 78.4983, "area": "Urban"},
        "Tarnaka": {"lat": 17.4278, "lon": 78.5354, "area": "Urban"},
        "Uppal": {"lat": 17.3984, "lon": 78.5583, "area": "Suburban"},
        "Dilsukhnagar": {"lat": 17.3687, "lon": 78.5247, "area": "Urban"},
        "LB Nagar": {"lat": 17.3497, "lon": 78.5524, "area": "Suburban"},
        "Koti": {"lat": 17.3854, "lon": 78.4867, "area": "Urban"},
        "Abids": {"lat": 17.3928, "lon": 78.4772, "area": "Urban"},
        "Mehdipatnam": {"lat": 17.3942, "lon": 78.4294, "area": "Urban"},
        "Tolichowki": {"lat": 17.4014, "lon": 78.4068, "area": "Urban"},
        "Gachibowli": {"lat": 17.4401, "lon": 78.3489, "area": "Suburban"},
        "Hitech City": {"lat": 17.4483, "lon": 78.3815, "area": "Urban"},
        "Madhapur": {"lat": 17.4486, "lon": 78.3908, "area": "Urban"},
        "Kondapur": {"lat": 17.4678, "lon": 78.3678, "area": "Suburban"},
        "Jubilee Hills": {"lat": 17.4239, "lon": 78.4077, "area": "Urban"},
        "Banjara Hills": {"lat": 17.4126, "lon": 78.4484, "area": "Urban"},
    }

    corridors: list[tuple[str, str, str, int, float]] = [
        ("Miyapur", "Kukatpally", "Highway", 80, 0.86),
        ("Miyapur", "Kondapur", "Arterial", 60, 0.48),
        ("Kondapur", "Gachibowli", "Arterial", 50, 0.34),
        ("Kondapur", "Hitech City", "Arterial", 50, 0.41),
        ("Gachibowli", "Hitech City", "Arterial", 50, 0.37),
        ("Gachibowli", "Tolichowki", "Collector", 50, 0.29),
        ("Hitech City", "Madhapur", "Arterial", 50, 0.44),
        ("Madhapur", "Jubilee Hills", "Arterial", 50, 0.33),
        ("Madhapur", "Kukatpally", "Collector", 50, 0.52),
        ("Kukatpally", "Moosapet", "Highway", 80, 0.91),
        ("Moosapet", "SR Nagar", "Arterial", 60, 0.72),
        ("SR Nagar", "Ameerpet", "Arterial", 50, 0.68),
        ("Ameerpet", "Panjagutta", "Arterial", 50, 0.64),
        ("Ameerpet", "Begumpet", "Collector", 40, 0.46),
        ("Panjagutta", "Begumpet", "Arterial", 50, 0.58),
        ("Panjagutta", "Banjara Hills", "Arterial", 40, 0.31),
        ("Begumpet", "Secunderabad", "Arterial", 50, 0.77),
        ("Secunderabad", "Tarnaka", "Arterial", 50, 0.61),
        ("Tarnaka", "Uppal", "Highway", 80, 0.84),
        ("Uppal", "Dilsukhnagar", "Arterial", 60, 0.79),
        ("Dilsukhnagar", "LB Nagar", "Arterial", 60, 0.74),
        ("Dilsukhnagar", "Koti", "Arterial", 50, 0.81),
        ("LB Nagar", "Koti", "Collector", 50, 0.57),
        ("Koti", "Abids", "Local", 40, 0.88),
        ("Abids", "Mehdipatnam", "Arterial", 40, 0.55),
        ("Abids", "Banjara Hills", "Collector", 40, 0.36),
        ("Mehdipatnam", "Tolichowki", "Arterial", 40, 0.32),
        ("Mehdipatnam", "Banjara Hills", "Collector", 40, 0.30),
        ("Tolichowki", "Jubilee Hills", "Arterial", 50, 0.27),
        ("Jubilee Hills", "Banjara Hills", "Local", 40, 0.22),
        ("Jubilee Hills", "Panjagutta", "Collector", 40, 0.28),
        ("Hitech City", "Jubilee Hills", "Collector", 50, 0.26),
        ("SR Nagar", "Madhapur", "Collector", 50, 0.49),
        ("Koti", "Banjara Hills", "Arterial", 40, 0.42),
        ("Secunderabad", "Koti", "Arterial", 50, 0.83),
        ("Gachibowli", "Miyapur", "Highway", 80, 0.80),
        ("Moosapet", "Begumpet", "Collector", 50, 0.70),
        ("Tarnaka", "Koti", "Collector", 50, 0.66),
        ("Jubilee Hills", "Abids", "Local", 40, 0.25),
        ("Tolichowki", "Banjara Hills", "Collector", 40, 0.28),
        ("Banjara Hills", "Begumpet", "Local", 40, 0.35),
        ("Madhapur", "Panjagutta", "Collector", 50, 0.40),
        ("Banjara Hills", "Secunderabad", "Collector", 40, 0.30),
        ("Jubilee Hills", "Begumpet", "Local", 40, 0.28),
        ("Tolichowki", "Abids", "Local", 40, 0.27),
    ]

    graph = nx.Graph()
    for node_id, meta in nodes.items():
        graph.add_node(node_id, **meta)

    junction_cycle = ["Cross", "T-Junction", "Roundabout", "Flyover", "Cross"]
    for index, (u, v, road_type, speed_limit, corridor_prior) in enumerate(corridors):
        lat1, lon1 = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
        lat2, lon2 = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
        distance = haversine_km(lat1, lon1, lat2, lon2)
        travel_time_min = (distance / max(speed_limit * 0.55, AVERAGE_CITY_SPEED_KMPH)) * 60.0
        graph.add_edge(
            u,
            v,
            distance_km=round(distance, 4),
            travel_time_min=round(travel_time_min, 3),
            road_type=road_type,
            speed_limit=speed_limit,
            corridor_prior=corridor_prior,
            junction_type=junction_cycle[index % len(junction_cycle)],
            mid_lat=round((lat1 + lat2) / 2.0, 6),
            mid_lon=round((lon1 + lon2) / 2.0, 6),
        )
    if not nx.is_connected(graph):
        raise RuntimeError("City graph must be a single connected component.")
    return graph


CITY_GRAPH = build_city_graph()
INTERSECTION_NAMES = sorted(CITY_GRAPH.nodes())


@lru_cache(maxsize=1)
def load_model_bundle(model_path: str = str(MODEL_PATH)) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run train_model.py before routing."
        )
    return joblib.load(path)


def predict_edge_risk(
    pipeline: Any,
    edge_data: dict[str, Any],
    *,
    weather: str,
    hour: int,
    day_of_week: str,
    month: int,
    traffic_density: float,
    lighting: str | None = None,
    temperature_c: float = 29.0,
    humidity: float = 62.0,
) -> float:
    resolved_lighting = lighting or lighting_from_hour(hour)
    row = build_inference_row(
        weather=weather,
        lighting=resolved_lighting,
        road_surface=surface_from_weather(weather),
        road_type=edge_data["road_type"],
        junction_type=edge_data["junction_type"],
        urban_rural="Urban",
        day_of_week=day_of_week,
        hour=int(hour),
        month=int(month),
        speed_limit=float(edge_data["speed_limit"]),
        traffic_density=float(traffic_density),
        visibility_km=visibility_from_weather(weather),
        precipitation_mm=precipitation_from_weather(weather),
        temperature_c=temperature_c,
        humidity=humidity,
        vehicle_count=2,
        latitude=float(edge_data["mid_lat"]),
        longitude=float(edge_data["mid_lon"]),
    )
    score = float(pipeline.predict(row)[0])
    return float(min(max(score, 0.0), 1.0))


def annotate_graph_risks(
    graph: nx.Graph,
    *,
    weather: str,
    hour: int,
    day_of_week: str,
    month: int,
    traffic_density: float,
    lighting: str | None = None,
    temperature_c: float = 29.0,
    humidity: float = 62.0,
    model_path: str = str(MODEL_PATH),
) -> nx.Graph:
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]
    annotated = graph.copy()
    for u, v, data in annotated.edges(data=True):
        model_risk = predict_edge_risk(
            pipeline,
            data,
            weather=weather,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            traffic_density=traffic_density,
            lighting=lighting,
            temperature_c=temperature_c,
            humidity=humidity,
        )
        prior = float(data.get("corridor_prior", 0.45))
        risk = min(max(0.55 * model_risk + 0.45 * prior, 0.02), 0.98)
        data["model_risk"] = model_risk
        data["predicted_risk"] = risk
        data["fast_cost"] = data["travel_time_min"]
        penalty = 1.0 + RISK_PENALTY_ALPHA * (risk ** RISK_PENALTY_BETA)
        data["safe_cost"] = data["distance_km"] * penalty
    return annotated


def _path_to_result(graph: nx.Graph, node_ids: list[str]) -> RouteResult:
    coordinates = [(graph.nodes[n]["lat"], graph.nodes[n]["lon"]) for n in node_ids]
    edges: list[dict[str, Any]] = []
    distance = 0.0
    travel = 0.0
    risks: list[float] = []
    for u, v in zip(node_ids[:-1], node_ids[1:]):
        data = graph[u][v]
        risk = float(data.get("predicted_risk", 0.0))
        distance += float(data["distance_km"])
        travel += float(data["travel_time_min"])
        risks.append(risk)
        edges.append(
            {
                "from": u,
                "to": v,
                "distance_km": float(data["distance_km"]),
                "travel_time_min": float(data["travel_time_min"]),
                "predicted_risk": risk,
                "road_type": data["road_type"],
            }
        )
    mean_risk = float(sum(risks) / len(risks)) if risks else 0.0
    max_risk = float(max(risks)) if risks else 0.0
    return RouteResult(
        node_ids=node_ids,
        coordinates=coordinates,
        edge_records=edges,
        distance_km=round(distance, 3),
        travel_time_min=round(travel, 2),
        mean_risk=round(mean_risk, 4),
        max_risk=round(max_risk, 4),
        cumulative_risk=round(float(sum(risks)), 4),
        safety_index=round(float(max(0.0, 1.0 - mean_risk)), 4),
    )


def _shortest_path(graph: nx.Graph, source: str, target: str, weight: str) -> list[str]:
    if source not in graph or target not in graph:
        raise KeyError("Unknown intersection name.")
    if source == target:
        return [source]

    def heuristic(u: str, v: str) -> float:
        u_data, v_data = graph.nodes[u], graph.nodes[v]
        geo = haversine_km(u_data["lat"], u_data["lon"], v_data["lat"], v_data["lon"])
        if weight == "fast_cost":
            max_effective_speed = 80.0 * 0.55
            return (geo / max_effective_speed) * 60.0
        return geo

    return nx.astar_path(graph, source, target, heuristic=heuristic, weight=weight)


def compute_dual_routes(
    source: str,
    target: str,
    *,
    weather: str,
    hour: int,
    day_of_week: str,
    month: int = 10,
    traffic_density: float = 0.45,
    lighting: str | None = None,
    temperature_c: float = 29.0,
    humidity: float = 62.0,
    graph: nx.Graph | None = None,
) -> dict[str, RouteResult]:
    working = annotate_graph_risks(
        graph or CITY_GRAPH,
        weather=weather,
        hour=hour,
        day_of_week=day_of_week,
        month=month,
        traffic_density=traffic_density,
        lighting=lighting,
        temperature_c=temperature_c,
        humidity=humidity,
    )
    fast_nodes = _shortest_path(working, source, target, "fast_cost")
    safe_nodes = _shortest_path(working, source, target, "safe_cost")
    return {
        "fast": _path_to_result(working, fast_nodes),
        "safe": _path_to_result(working, safe_nodes),
    }


def graph_map_center(graph: nx.Graph = CITY_GRAPH) -> tuple[float, float]:
    lats = [data["lat"] for _, data in graph.nodes(data=True)]
    lons = [data["lon"] for _, data in graph.nodes(data=True)]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def edge_table(graph: nx.Graph | None = None) -> pd.DataFrame:
    g = graph or CITY_GRAPH
    rows = []
    for u, v, data in g.edges(data=True):
        rows.append(
            {
                "from": u,
                "to": v,
                "distance_km": data["distance_km"],
                "road_type": data["road_type"],
                "speed_limit": data["speed_limit"],
            }
        )
    return pd.DataFrame(rows).sort_values(["from", "to"]).reset_index(drop=True)
