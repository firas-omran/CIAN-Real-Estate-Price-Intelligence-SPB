"""Streamlit UI for CIAN Real Estate Price Intelligence."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.predictor import predict_price, serving_options
from src.monitoring.checkpoint4 import monitoring_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("streamlit")

API_URL = os.getenv("API_URL", "").rstrip("/")


def _call_predict_api(**kwargs: object) -> dict:
    """Call the FastAPI /predict endpoint, or fall back to in-process prediction.

    When API_URL is set (e.g. to the ngrok/bore public tunnel), an HTTP POST is
    sent so that Prometheus metrics, Kafka events and logs are recorded on the
    server side.  Without API_URL the function calls predict_price directly —
    this is the dev/local fallback.
    """
    if API_URL:
        payload = {k: v for k, v in kwargs.items() if v is not None and k not in ("street", "house_number")}
        payload.setdefault("street", "")
        payload.setdefault("house_number", "")
        started = time.perf_counter()
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        http_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        result = response.json()
        logger.info(
            "streamlit->api: model=%s district=%s rooms=%d area=%.0f price=%.0f http_ms=%.1f kafka=%s",
            result.get("model_name", "?"),
            kwargs.get("district", "?"),
            kwargs.get("rooms_count", -1),
            kwargs.get("total_meters", 0),
            result.get("predicted_price", 0),
            http_ms,
            result.get("kafka_event_published", False),
        )
        # Re-wrap features dict back from JSON (keys become plain strings)
        features = result.get("features", {})
        return {
            "model_name": result["model_name"],
            "predicted_price": result["predicted_price"],
            "predicted_price_per_sqm": result["predicted_price_per_sqm"],
            "price_range_low": result.get("price_range_low", result["predicted_price"] * 0.85),
            "price_range_high": result.get("price_range_high", result["predicted_price"] * 1.15),
            "features": features,
            "kafka_event_published": result.get("kafka_event_published", False),
        }

    # Fallback: direct in-process call (no FastAPI available)
    return predict_price(**{k: v for k, v in kwargs.items() if not k.startswith("_")})


def rub(value: float) -> str:
    return f"{value:,.0f} руб.".replace(",", " ")


st.set_page_config(
    page_title="CIAN Real Estate Price Intelligence",
    layout="wide",
)

st.title("CIAN Real Estate Price Intelligence")
st.markdown("ML-система для оценки стоимости квартир в Санкт-Петербурге по свежим объявлениям CIAN.")

options = serving_options()
predict_tab, monitoring_tab = st.tabs(["Prediction", "Monitoring & Drift"])

with predict_tab:
    st.sidebar.header("Параметры квартиры")

    rooms_count = st.sidebar.selectbox(
        "Количество комнат",
        options=[0, 1, 2, 3, 4],
        format_func=lambda value: "Студия" if value == 0 else str(value),
    )
    total_meters = st.sidebar.number_input("Площадь, м2", min_value=10.0, max_value=500.0, value=58.0, step=1.0)
    floor = st.sidebar.number_input("Этаж", min_value=1, max_value=100, value=5, step=1)
    floors_count = st.sidebar.number_input("Этажей в доме", min_value=1, max_value=100, value=12, step=1)
    district = st.sidebar.selectbox("Район", options=options["districts"])
    underground = st.sidebar.selectbox("Метро", options=options["underground"])
    author_type = st.sidebar.selectbox("Тип продавца", options=options["author_types"])

    street = st.sidebar.text_input("Улица (опционально)", value="")
    house_number = st.sidebar.text_input("Дом (опционально)", value="")

    if floor > floors_count:
        st.sidebar.warning("Этаж не должен быть больше количества этажей в доме.")

    if st.sidebar.button("Оценить стоимость", type="primary", disabled=floor > floors_count):
        started = time.perf_counter()
        result = _call_predict_api(
            rooms_count=rooms_count,
            total_meters=total_meters,
            floor=floor,
            floors_count=floors_count,
            district=district,
            underground=underground,
            author_type=author_type,
            street=street or None,
            house_number=house_number or None,
        )
        ui_latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "streamlit prediction: model=%s district=%s rooms=%d area=%.0f price=%.0f ui_latency_ms=%.1f kafka=%s",
            result["model_name"],
            district,
            rooms_count,
            total_meters,
            result["predicted_price"],
            ui_latency_ms,
            result.get("kafka_event_published", False),
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Предсказанная цена", rub(result["predicted_price"]))
        col2.metric("Цена за м2", rub(result["predicted_price_per_sqm"]))
        col3.metric("Модель", result["model_name"])

        low = rub(result["price_range_low"])
        high = rub(result["price_range_high"])
        st.info(f"Оценочный диапазон: {low} - {high}")

        features = result["features"]
        geo_cols = st.columns(3)
        geo_cols[0].metric("Гео-точность", str(features.get("geo_precision")))
        distance_center = features.get("distance_to_center_km")
        distance_metro = features.get("distance_to_metro_route_km")
        geo_cols[1].metric("До центра", f"{distance_center:.1f} км" if distance_center else "-")
        geo_cols[2].metric("До метро", f"{distance_metro:.1f} км" if distance_metro else "-")

        with st.expander("Model input features"):
            st.dataframe(features, use_container_width=True)
    else:
        st.info("Введите параметры квартиры и нажмите «Оценить стоимость».")

with monitoring_tab:
    scenario = st.radio(
        "Monitoring scenario",
        options=["normal", "drifted"],
        horizontal=True,
        format_func=lambda value: "Normal snapshot" if value == "normal" else "Degradation / drift demo",
    )
    snapshot = monitoring_snapshot(scenario)
    quality = snapshot["quality"]
    prediction = snapshot["prediction"]
    system = snapshot["system"]

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", quality["rows"])
    q2.metric("Snapshot age", f"{quality['snapshot_age_days']:.1f} days")
    q3.metric("Geocode coverage", f"{quality['geocode_coverage']:.1%}")
    q4.metric("Target missing", f"{quality['target_missing_share']:.1%}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", prediction["model_name"])
    m2.metric("Median prediction", rub(prediction["predicted_price_median"]))
    m3.metric("P95 prediction", rub(prediction["predicted_price_p95"]))
    m4.metric("API p95 latency", f"{system['p95_latency_ms']:.0f} ms")

    alerts = snapshot["alerts"]
    fired_alerts = [alert for alert in alerts if alert["fired"]]
    if fired_alerts:
        st.error(f"Active alerts: {len(fired_alerts)}")
    else:
        st.success("No active alerts")

    st.subheader("Alerts")
    if fired_alerts:
        st.markdown("Active alert actions:")
        st.table(
            [
                {
                    "alert": alert["name"],
                    "level": alert["level"],
                    "value": alert["value"],
                    "action": alert["action"],
                }
                for alert in fired_alerts
            ]
        )
    st.dataframe(
        [
            {
                "level": alert["level"],
                "alert": alert["name"],
                "fired": alert["fired"],
                "value": alert["value"],
                "threshold": alert["threshold"],
                "action": alert["action"],
            }
            for alert in alerts
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Data Drift")
    drift_table = snapshot["drift"]
    drifted_features = drift_table[drift_table["drift_detected"]].copy()
    if drifted_features.empty:
        st.success("No data drift detected against the reference training snapshot.")
    else:
        st.warning("Data drift detected. Features above threshold:")
        st.table(
            drifted_features[["feature", "type", "metric", "value", "threshold"]]
            .round({"value": 4})
            .to_dict("records")
        )
    st.dataframe(drift_table.round({"value": 4}), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("*Проект в рамках курса 'Архитектура ИИ' - ИТМО*")
