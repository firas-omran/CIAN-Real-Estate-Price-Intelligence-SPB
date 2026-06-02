"""Shared prediction utilities for FastAPI and Streamlit serving."""

from __future__ import annotations

import logging
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.api.telemetry import publish_prediction_event, record_prediction_metrics
from src.data.clean_cian import VALID_SPB_DISTRICTS
from src.features.geocoder import (
    distance_to_center_km,
    enrich_listing,
    geocode_freeform_address,
    nearest_district,
    nearest_metro,
)

logger = logging.getLogger(__name__)

METRICS_PATH = Path("data/experiments/checkpoint3_metrics.csv")
MODEL_DIR = Path("data/models")
FEATURES_PATH = Path("data/features/cian_spb_offline_features.csv")
CLEAN_DATA_PATH = Path("data/processed/cian_spb_clean.csv")
GEOCODE_CACHE_PATH = Path("data/cache/geocode_cache.json")
BASELINE_MODEL = "B2_non_ml_baseline"
HOUSE_NUMBER_PATTERN = re.compile(r"^\s*\d+")
ADDRESS_HAS_NUMBER_PATTERN = re.compile(r"\d")
ADDRESS_HAS_LETTER_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _room_segment(rooms_count: int) -> str:
    if rooms_count <= 0:
        return "studio"
    if rooms_count == 1:
        return "1room"
    return f"{rooms_count}rooms"


@lru_cache(maxsize=1)
def load_reference_data() -> pd.DataFrame:
    """Load the offline feature table used for UI options and fallbacks."""
    return pd.read_csv(FEATURES_PATH)


@lru_cache(maxsize=1)
def load_best_model() -> tuple[str, Any]:
    """Load the best saved sklearn model according to latest experiment metrics."""
    metrics = pd.read_csv(METRICS_PATH)
    model_rows = metrics[metrics["model"] != BASELINE_MODEL].copy()
    if model_rows.empty:
        raise RuntimeError("No trained ML models found in checkpoint3 metrics.")

    best = model_rows.sort_values("test_r2_per_sqm", ascending=False).iloc[0]
    model_name = str(best["model"])
    model_path = MODEL_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model file is missing: {model_path}")
    return model_name, joblib.load(model_path)


def serving_options() -> dict[str, list[str]]:
    """Return stable option lists for Streamlit controls."""
    df = load_reference_data()
    options = {
        "districts": sorted(df["district"].dropna().astype(str).unique().tolist()),
        "underground": sorted(df["underground"].dropna().astype(str).unique().tolist()),
        "author_types": sorted(df["author_type"].dropna().astype(str).unique().tolist()),
    }
    options["address_examples"] = _address_examples()
    return options


@lru_cache(maxsize=1)
def _address_examples(limit: int = 12) -> list[str]:
    """Return real address examples from the cleaned CIAN snapshot."""
    examples: list[str] = []
    if GEOCODE_CACHE_PATH.exists():
        try:
            cache = json.loads(GEOCODE_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
        for key, value in cache.items():
            if not key.startswith("house|"):
                continue
            try:
                latlon = (float(value["lat"]), float(value["lon"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not (59.73 <= latlon[0] <= 60.15 and 29.65 <= latlon[1] <= 30.70):
                continue
            parts = key.split("|")
            if len(parts) != 4 or not HOUSE_NUMBER_PATTERN.match(parts[3]):
                continue
            examples.append(f"{parts[2].title()}, {parts[3]}")
            if len(examples) >= limit:
                return examples

    if not CLEAN_DATA_PATH.exists():
        return examples
    df = pd.read_csv(CLEAN_DATA_PATH, usecols=["street", "house_number"])
    df = df.dropna()
    df["street"] = df["street"].astype(str).str.strip()
    df["house_number"] = df["house_number"].astype(str).str.strip()
    df = df[(df["street"] != "") & df["house_number"].str.match(HOUSE_NUMBER_PATTERN)]
    examples.extend(f"{row.street}, {row.house_number}" for row in df.drop_duplicates().head(limit).itertuples())
    return examples[:limit]


def _has_address(street: str | None, house_number: str | None) -> bool:
    return bool(str(street or "").strip() or str(house_number or "").strip())


def _validate_address(street: str | None, house_number: str | None) -> None:
    if not _has_address(street, house_number):
        return
    if not str(street or "").strip():
        raise ValueError("Введите адрес квартиры.")
    if ADDRESS_HAS_LETTER_PATTERN.search(str(street or "")) is None:
        raise ValueError("Введите адрес или улицу, а не только цифры.")
    if str(house_number or "").strip() and HOUSE_NUMBER_PATTERN.match(str(house_number or "")) is None:
        raise ValueError("Номер дома должен начинаться с цифры, например: 28, 10к2, 7литА.")


def _resolve_serving_geo(
    *,
    district: str | None,
    underground: str | None,
    street: str | None,
    house_number: str | None,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve serving-time geo features from address or manual fallback."""
    _validate_address(street, house_number)

    address_result = None
    if _has_address(street, house_number):
        address_result = geocode_freeform_address(street=street, house_number=house_number)
        if address_result is None:
            raise ValueError(
                "Адрес не найден внутри границ Санкт-Петербурга. "
                "Выберите подсказку 2GIS или уточните улицу и номер дома."
            )

    if address_result is not None:
        inferred_district = nearest_district(address_result.lat, address_result.lon)
        if inferred_district is None:
            raise ValueError("Не удалось определить район по адресу.")

        nearest_station = nearest_metro(address_result.lat, address_result.lon)
        inferred_underground = nearest_station[0] if nearest_station else "unknown"
        distance_metro = nearest_station[1] if nearest_station else None
        distance_center = distance_to_center_km(address_result.lat, address_result.lon)
        duration_metro = float(distance_metro) / 5.0 * 60.0 if distance_metro is not None else None

        return inferred_district, inferred_underground, {
            "lat": address_result.lat,
            "lon": address_result.lon,
            "geo_precision": address_result.precision,
            "distance_to_center_km": distance_center,
            "distance_to_metro_km": distance_metro,
            "distance_to_metro_route_km": distance_metro,
            "duration_to_metro_route_min": duration_metro,
            "metro_route_provider": "nearest_metro:haversine",
            "metro_known": nearest_station is not None,
            "address_mode": "address",
        }

    fallback_district = str(district or "").strip()
    if fallback_district not in VALID_SPB_DISTRICTS:
        raise ValueError("Выберите район или введите полный адрес.")
    fallback_underground = str(underground or "unknown").strip() or "unknown"
    geo = enrich_listing(
        district=fallback_district,
        street=None,
        house_number=None,
        underground=fallback_underground,
        with_routing=False,
    )
    geo["address_mode"] = "district_fallback"
    return fallback_district, fallback_underground, geo


def make_feature_row(
    *,
    rooms_count: int,
    total_meters: float,
    floor: int,
    floors_count: int,
    district: str | None = None,
    underground: str = "unknown",
    author_type: str = "real_estate_agent",
    street: str | None = None,
    house_number: str | None = None,
) -> pd.DataFrame:
    """Build one inference row with the feature names used by checkpoint 3."""
    floor_ratio = floor / floors_count if floors_count else np.nan
    resolved_district, resolved_underground, geo = _resolve_serving_geo(
        district=district,
        street=street,
        house_number=house_number,
        underground=underground,
    )

    route_distance = geo.get("distance_to_metro_route_km")
    route_duration = geo.get("duration_to_metro_route_min")
    route_provider = geo.get("metro_route_provider")
    if route_distance is None:
        route_distance = geo.get("distance_to_metro_km")
    if route_duration is None and route_distance is not None:
        route_duration = float(route_distance) / 5.0 * 60.0
    if route_provider is None and route_distance is not None:
        route_provider = "haversine_fallback"

    row = {
        "rooms_count": rooms_count,
        "total_meters": total_meters,
        "log_total_meters": np.log1p(total_meters),
        "floor": floor,
        "floors_count": floors_count,
        "floor_ratio": floor_ratio,
        "is_first_floor": int(floor == 1),
        "is_last_floor": int(floor == floors_count),
        "lat": geo.get("lat"),
        "lon": geo.get("lon"),
        "distance_to_center_km": geo.get("distance_to_center_km"),
        "distance_to_metro_route_km": route_distance,
        "duration_to_metro_route_min": route_duration,
        "metro_known": bool(geo.get("metro_known")),
        "author_type": author_type,
        "room_segment": _room_segment(rooms_count),
        "district": resolved_district,
        "underground": resolved_underground,
        "geo_precision": geo.get("geo_precision") or "district",
        "metro_route_provider": route_provider or "unknown",
        "address_mode": geo.get("address_mode", "district_fallback"),
    }
    return pd.DataFrame([row])


def predict_price(**kwargs: Any) -> dict[str, Any]:
    """Predict price per square meter and reconstructed apartment price.

    Emits Prometheus metrics and publishes a Kafka event when the required
    packages and environment variables are available.  Both the FastAPI
    endpoint and the Streamlit UI share this function, so telemetry is
    recorded regardless of the entry point.
    """
    started = time.perf_counter()
    model_name, model = load_best_model()
    features = make_feature_row(**kwargs)
    pred_log = float(model.predict(features)[0])
    price_per_sqm = float(np.expm1(pred_log))
    total_meters = float(features.loc[0, "total_meters"])
    predicted_price = price_per_sqm * total_meters
    latency_ms = (time.perf_counter() - started) * 1000

    # --- telemetry (no-op when packages/env are missing) ---
    record_prediction_metrics(model_name, predicted_price, latency_ms=latency_ms, status="ok")
    district = str(features.loc[0, "district"]) if "district" in features.columns else "unknown"
    kafka_event_published = publish_prediction_event(
        {
            "model_name": model_name,
            "predicted_price": predicted_price,
            "predicted_price_per_sqm": price_per_sqm,
            "district": district,
            "rooms_count": int(features.loc[0, "rooms_count"]) if "rooms_count" in features.columns else -1,
            "total_meters": total_meters,
            "latency_ms": round(latency_ms, 2),
        }
    )

    logger.info(
        "prediction: model=%s district=%s rooms=%d area=%.0f price=%.0f price_per_sqm=%.0f latency_ms=%.1f kafka=%s",
        model_name,
        district,
        int(features.loc[0, "rooms_count"]) if "rooms_count" in features.columns else -1,
        total_meters,
        predicted_price,
        price_per_sqm,
        latency_ms,
        kafka_event_published,
    )

    return {
        "model_name": model_name,
        "predicted_price": predicted_price,
        "predicted_price_per_sqm": price_per_sqm,
        "price_range_low": predicted_price * 0.85,
        "price_range_high": predicted_price * 1.15,
        "features": features.iloc[0].to_dict(),
        "kafka_event_published": kafka_event_published,
    }
