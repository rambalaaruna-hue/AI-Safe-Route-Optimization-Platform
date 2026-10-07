"""Streamlit control room for risk-aware dual-path routing."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config import DAY_VALUES, MODEL_PATH, WEATHER_VALUES
from routing_engine import (
    CITY_GRAPH,
    INTERSECTION_NAMES,
    compute_dual_routes,
    graph_map_center,
    lighting_from_hour,
)

st.set_page_config(
    page_title="Safe Route Optimization Platform",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 1400px;}
    div[data-testid="stMetric"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
    }
    div[data-testid="stMetric"] label {color: #94a3b8 !important;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #f8fafc !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _model_ready() -> bool:
    return Path(MODEL_PATH).is_file()


def _draw_route(fmap: folium.Map, coordinates: list[tuple[float, float]], color: str, label: str) -> None:
    if len(coordinates) < 2:
        return
    folium.PolyLine(
        locations=coordinates,
        color=color,
        weight=7,
        opacity=0.92,
        tooltip=label,
    ).add_to(fmap)
    folium.Marker(
        coordinates[0],
        tooltip="Origin",
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
    ).add_to(fmap)
    folium.Marker(
        coordinates[-1],
        tooltip="Destination",
        icon=folium.Icon(color="purple", icon="flag", prefix="fa"),
    ).add_to(fmap)


def _comparison_frame(fast, safe) -> pd.DataFrame:
    extra_distance = 0.0
    if fast.distance_km > 0:
        extra_distance = 100.0 * (safe.distance_km - fast.distance_km) / fast.distance_km
    risk_drop = 0.0
    if fast.mean_risk > 0:
        risk_drop = 100.0 * (fast.mean_risk - safe.mean_risk) / fast.mean_risk
    return pd.DataFrame(
        {
            "Metric": [
                "Distance (km)",
                "Travel time (min)",
                "Mean accident risk",
                "Peak corridor risk",
                "Safety index",
                "Cumulative risk",
                "Intersections traversed",
            ],
            "High-risk fast path": [
                f"{fast.distance_km:.2f}",
                f"{fast.travel_time_min:.1f}",
                f"{fast.mean_risk:.3f}",
                f"{fast.max_risk:.3f}",
                f"{fast.safety_index:.3f}",
                f"{fast.cumulative_risk:.3f}",
                str(max(len(fast.node_ids) - 1, 0)),
            ],
            "AI safest path": [
                f"{safe.distance_km:.2f}",
                f"{safe.travel_time_min:.1f}",
                f"{safe.mean_risk:.3f}",
                f"{safe.max_risk:.3f}",
                f"{safe.safety_index:.3f}",
                f"{safe.cumulative_risk:.3f}",
                str(max(len(safe.node_ids) - 1, 0)),
            ],
            "Delta": [
                f"{extra_distance:+.1f}% distance",
                f"{safe.travel_time_min - fast.travel_time_min:+.1f} min",
                f"{risk_drop:+.1f}% risk",
                f"{safe.max_risk - fast.max_risk:+.3f}",
                f"{safe.safety_index - fast.safety_index:+.3f}",
                f"{safe.cumulative_risk - fast.cumulative_risk:+.3f}",
                f"{(len(safe.node_ids) - len(fast.node_ids)):+d} nodes",
            ],
        }
    )


st.title("Multi-Variant Heuristic Risk Modeling")
st.caption(
    "Dynamic spatio-temporal safe-route optimization for Hyderabad using XGBoost risk scores "
    "and A* search with a risk-penalized edge cost."
)

if not _model_ready():
    st.error(
        "The trained model `models/accident_model.pkl` is missing. "
        "Run `python train_model.py` or `run_train.bat` first, then refresh this page."
    )
    st.stop()

with st.sidebar:
    st.header("Trip context")
    origin = st.selectbox("Origin intersection", INTERSECTION_NAMES, index=INTERSECTION_NAMES.index("Gachibowli"))
    destination = st.selectbox(
        "Destination intersection",
        INTERSECTION_NAMES,
        index=INTERSECTION_NAMES.index("Secunderabad"),
    )
    st.divider()
    st.header("Weather")
    weather = st.selectbox("Atmospheric condition", WEATHER_VALUES, index=2)
    temperature_c = st.slider("Temperature (°C)", 16, 44, 29)
    humidity = st.slider("Humidity (%)", 20, 98, 64)
    st.divider()
    st.header("Time")
    hour = st.slider("Hour of day", 0, 23, 21)
    day_of_week = st.selectbox("Day of week", DAY_VALUES, index=4)
    month = st.slider("Month", 1, 12, 10)
    auto_light = lighting_from_hour(hour)
    lighting = st.selectbox(
        "Lighting",
        ["Daylight", "Dusk", "Night Lit", "Night Unlit"],
        index=["Daylight", "Dusk", "Night Lit", "Night Unlit"].index(auto_light),
    )
    st.divider()
    st.header("Traffic")
    traffic_density = st.slider("Congestion intensity", 0.05, 0.98, 0.62, 0.01)
    st.caption("0 = free flow, 1 = standstill. This value is injected into every edge feature vector.")
    compute = st.button("Compute dual routes", type="primary", use_container_width=True)

if origin == destination:
    st.warning("Choose two different intersections to generate a route pair.")
    st.stop()

if compute or "routes" not in st.session_state:
    with st.spinner("Scoring corridors with XGBoost and running A* on the city graph..."):
        st.session_state["routes"] = compute_dual_routes(
            origin,
            destination,
            weather=weather,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            traffic_density=traffic_density,
            lighting=lighting,
            temperature_c=temperature_c,
            humidity=humidity,
        )
        st.session_state["context"] = {
            "origin": origin,
            "destination": destination,
            "weather": weather,
            "hour": hour,
            "temperature_c": temperature_c,
            "humidity": humidity,
        }

routes = st.session_state["routes"]
fast = routes["fast"]
safe = routes["safe"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fast path distance", f"{fast.distance_km:.2f} km")
k2.metric("Safest path distance", f"{safe.distance_km:.2f} km")
k3.metric("Fast path mean risk", f"{fast.mean_risk:.3f}")
k4.metric("Safest path mean risk", f"{safe.mean_risk:.3f}")

center = graph_map_center(CITY_GRAPH)
fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

for node, data in CITY_GRAPH.nodes(data=True):
    folium.CircleMarker(
        location=(data["lat"], data["lon"]),
        radius=4,
        color="#334155",
        fill=True,
        fill_opacity=0.7,
        tooltip=node,
    ).add_to(fmap)

_draw_route(fmap, fast.coordinates, "#dc2626", "High-risk fast path")
_draw_route(fmap, safe.coordinates, "#16a34a", "AI-recommended safest path")

legend_html = """
<div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white;
padding: 10px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 13px;">
<b>Legend</b><br>
<span style="color:#dc2626;">━</span> High-risk fast path<br>
<span style="color:#16a34a;">━</span> AI safest path
</div>
"""
fmap.get_root().html.add_child(folium.Element(legend_html))

map_col, table_col = st.columns([1.35, 1.0], gap="large")
with map_col:
    st.subheader("Interactive corridor map")
    st_folium(fmap, width=None, height=560, returned_objects=[], use_container_width=True)
    st.caption(f"{origin} → {destination}  ·  {weather}  ·  {day_of_week} {hour:02d}:00  ·  lighting {lighting}")

with table_col:
    st.subheader("Distance vs safety analytics")
    st.dataframe(_comparison_frame(fast, safe), hide_index=True, use_container_width=True)
    st.markdown("**Fast path**")
    st.write(" → ".join(fast.node_ids))
    st.markdown("**Safest path**")
    st.write(" → ".join(safe.node_ids))

left, right = st.columns(2)
with left:
    st.markdown("#### Fast path segments")
    st.dataframe(pd.DataFrame(fast.edge_records), hide_index=True, use_container_width=True)
with right:
    st.markdown("#### Safest path segments")
    st.dataframe(pd.DataFrame(safe.edge_records), hide_index=True, use_container_width=True)
