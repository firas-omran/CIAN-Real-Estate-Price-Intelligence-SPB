"""Unit tests for geo helpers in src.features.geocoder."""

from __future__ import annotations

import math

from src.features.geocoder import (
    CENTER_LAT,
    CENTER_LON,
    distance_to_center_km,
    haversine_km,
)


def test_haversine_zero_distance() -> None:
    assert haversine_km(59.9386, 30.3141, 59.9386, 30.3141) == 0.0


def test_haversine_msk_to_spb() -> None:
    moscow_lat, moscow_lon = 55.7558, 37.6173
    spb_lat, spb_lon = 59.9386, 30.3141
    distance = haversine_km(moscow_lat, moscow_lon, spb_lat, spb_lon)
    # Reference distance Moscow ↔ Saint Petersburg ≈ 634 km
    assert 620.0 <= distance <= 650.0


def test_haversine_symmetric() -> None:
    a = haversine_km(59.93, 30.30, 60.00, 30.40)
    b = haversine_km(60.00, 30.40, 59.93, 30.30)
    assert math.isclose(a, b, rel_tol=1e-9)


def test_distance_to_center_zero() -> None:
    assert distance_to_center_km(CENTER_LAT, CENTER_LON) == 0.0


def test_distance_to_center_positive() -> None:
    nearby_lat, nearby_lon = 59.95, 30.40
    distance = distance_to_center_km(nearby_lat, nearby_lon)
    assert distance > 0
    assert distance < 10
