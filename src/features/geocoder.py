"""Geocoding for CIAN listings: address and metro to (lat, lon) and distances.

Provides a single reusable surface used both by the offline pipeline (when
preparing the training feature table) and by the future serving API (when
the user submits a free-form address).

The module talks to OpenStreetMap Nominatim through `geopy`, respects its
rate limit (1 request per second), and persists every successful lookup
to a JSON cache so subsequent runs are deterministic and offline-friendly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from src.data.clean_cian import VALID_SPB_DISTRICTS

CITY = "Санкт-Петербург"
COUNTRY = "Россия"
CENTER_LAT = 59.9386
CENTER_LON = 30.3141

USER_AGENT = "cian-spb-price-intelligence/0.1 (educational; contact: kurzo on github)"
MIN_DELAY_SECONDS = 1.0
MAX_RETRIES = 3

CACHE_DIR = Path("data/cache")
REFERENCE_DIR = Path("data/reference")
GEOCODE_CACHE_PATH = CACHE_DIR / "geocode_cache.json"
METRO_COORDS_PATH = REFERENCE_DIR / "metro_spb_coords.json"
DISTRICT_CENTROIDS_PATH = REFERENCE_DIR / "spb_district_centroids.json"

UNKNOWN_VALUES = {"unknown", "", "nan", "none"}

METRO_SEED_COORDS: dict[str, dict[str, float]] = {
    "Девяткино": {"lat": 60.0498, "lon": 30.4427},
    "Маяковская": {"lat": 59.9319, "lon": 30.3552},
    "Достоевская": {"lat": 59.9279, "lon": 30.3489},
    "Зенит": {"lat": 59.9716, "lon": 30.2089},
}


@dataclass(frozen=True)
class GeocodeResult:
    lat: float
    lon: float
    precision: str  # "house" | "street" | "district"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    earth_radius_km = 6371.0088
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    d_lat = lat2_rad - lat1_rad
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def distance_to_center_km(lat: float, lon: float) -> float:
    """Haversine distance from (lat, lon) to Saint Petersburg center (Дворцовая)."""
    return haversine_km(lat, lon, CENTER_LAT, CENTER_LON)


def _is_unknown(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in UNKNOWN_VALUES


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _make_address_key(tier: str, district: str, street: str, house: str) -> str:
    parts = [tier, district.strip().lower(), street.strip().lower(), house.strip().lower()]
    return "|".join(parts)


class _Geolocator:
    """Lazy singleton wrapper over geopy.Nominatim with rate limiting and retries."""

    _instance: Optional["_Geolocator"] = None

    def __init__(self) -> None:
        self.nominatim = Nominatim(user_agent=USER_AGENT, timeout=10)
        self.geocode = RateLimiter(
            self.nominatim.geocode,
            min_delay_seconds=MIN_DELAY_SECONDS,
            max_retries=MAX_RETRIES,
            error_wait_seconds=2.0,
            swallow_exceptions=False,
        )

    @classmethod
    def instance(cls) -> "_Geolocator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def lookup(self, query: str | dict) -> tuple[float, float] | None:
        for attempt in range(MAX_RETRIES):
            try:
                location = self.geocode(query, country_codes="ru", addressdetails=False, language="ru")
                if location is None:
                    return None
                return float(location.latitude), float(location.longitude)
            except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(2.0 * (attempt + 1))
        return None


# --- Public API ---------------------------------------------------------------


def geocode_address(
    district: str | None,
    street: str | None,
    house_number: str | None,
    cache: dict | None = None,
) -> GeocodeResult | None:
    """Return geocoded address using three precision tiers.

    Tries `house` → `street` → `district` (centroid). Returns None only if all
    three tiers fail, which should not happen because district centroids are
    precomputed for every entry in VALID_SPB_DISTRICTS.
    """
    if cache is None:
        cache = _load_json(GEOCODE_CACHE_PATH)

    district_clean = "" if _is_unknown(district) else str(district).strip()
    street_clean = "" if _is_unknown(street) else str(street).strip()
    house_clean = "" if _is_unknown(house_number) else str(house_number).strip()

    if district_clean not in VALID_SPB_DISTRICTS:
        return None

    geolocator = _Geolocator.instance()

    if house_clean and street_clean:
        key = _make_address_key("house", district_clean, street_clean, house_clean)
        cached = cache.get(key)
        if cached:
            return GeocodeResult(cached["lat"], cached["lon"], "house")
        query = {"street": f"{street_clean} {house_clean}", "city": CITY}
        latlon = geolocator.lookup(query)
        if latlon is not None and _within_spb_bbox(latlon):
            cache[key] = {"lat": latlon[0], "lon": latlon[1]}
            _save_json(cache, GEOCODE_CACHE_PATH)
            return GeocodeResult(latlon[0], latlon[1], "house")

    if street_clean:
        key = _make_address_key("street", district_clean, street_clean, "")
        cached = cache.get(key)
        if cached:
            return GeocodeResult(cached["lat"], cached["lon"], "street")
        query = {"street": street_clean, "city": CITY}
        latlon = geolocator.lookup(query)
        if latlon is not None and _within_spb_bbox(latlon):
            cache[key] = {"lat": latlon[0], "lon": latlon[1]}
            _save_json(cache, GEOCODE_CACHE_PATH)
            return GeocodeResult(latlon[0], latlon[1], "street")

    centroids = _load_district_centroids_or_seed()
    centroid = centroids.get(district_clean)
    if centroid:
        return GeocodeResult(centroid["lat"], centroid["lon"], "district")

    return None


def geocode_metro(station_name: str | None, cache: dict | None = None) -> tuple[float, float] | None:
    """Resolve a metro station name in Saint Petersburg to (lat, lon)."""
    if _is_unknown(station_name):
        return None
    name = str(station_name).strip()
    if cache is None:
        cache = _load_json(METRO_COORDS_PATH)
    cached = cache.get(name)
    if cached:
        return float(cached["lat"]), float(cached["lon"])

    seed = METRO_SEED_COORDS.get(name)
    if seed is not None:
        cache[name] = {"lat": seed["lat"], "lon": seed["lon"]}
        _save_json(cache, METRO_COORDS_PATH)
        return seed["lat"], seed["lon"]

    geolocator = _Geolocator.instance()
    query = f"станция метро {name}, {CITY}"
    latlon = geolocator.lookup(query)
    if latlon is None or not _within_spb_bbox(latlon):
        return None

    cache[name] = {"lat": latlon[0], "lon": latlon[1]}
    _save_json(cache, METRO_COORDS_PATH)
    return latlon


def _load_district_centroids_or_seed() -> dict:
    """Return district centroids; geocode any missing district name on first call."""
    centroids = _load_json(DISTRICT_CENTROIDS_PATH)
    missing = [d for d in VALID_SPB_DISTRICTS if d not in centroids]
    if not missing:
        return centroids

    geolocator = _Geolocator.instance()
    for district in missing:
        query = f"{district} район, {CITY}"
        latlon = geolocator.lookup(query)
        if latlon is None:
            continue
        centroids[district] = {"lat": latlon[0], "lon": latlon[1]}
    _save_json(centroids, DISTRICT_CENTROIDS_PATH)
    return centroids


def _within_spb_bbox(latlon: tuple[float, float]) -> bool:
    """Reject geocoder hits that landed far outside Saint Petersburg agglomeration.

    SPB plus Leningrad Oblast neighborhoods fit comfortably inside this bbox.
    Used as a sanity filter against ambiguous street names that resolve to
    other cities.
    """
    lat, lon = latlon
    return 59.5 <= lat <= 60.3 and 29.4 <= lon <= 30.9


def enrich_listing(
    district: str | None,
    street: str | None,
    house_number: str | None,
    underground: str | None,
    geocode_cache: dict | None = None,
    metro_cache: dict | None = None,
) -> dict:
    """Compute geo features for a single listing or an API request.

    Returns a dict with keys: lat, lon, geo_precision,
    distance_to_center_km, distance_to_metro_km, metro_known.
    """
    address_result = geocode_address(district, street, house_number, cache=geocode_cache)
    if address_result is None:
        return {
            "lat": None,
            "lon": None,
            "geo_precision": None,
            "distance_to_center_km": None,
            "distance_to_metro_km": None,
            "metro_known": False,
        }

    distance_center = distance_to_center_km(address_result.lat, address_result.lon)

    metro_known = not _is_unknown(underground)
    distance_metro: float | None = None
    if metro_known:
        metro_latlon = geocode_metro(underground, cache=metro_cache)
        if metro_latlon is not None:
            distance_metro = haversine_km(
                address_result.lat, address_result.lon, metro_latlon[0], metro_latlon[1]
            )
        else:
            metro_known = False

    return {
        "lat": address_result.lat,
        "lon": address_result.lon,
        "geo_precision": address_result.precision,
        "distance_to_center_km": distance_center,
        "distance_to_metro_km": distance_metro,
        "metro_known": metro_known,
    }


# --- CLI: enrich a clean snapshot ---------------------------------------------


def enrich_dataframe(df: pd.DataFrame, progress_every: int = 50) -> pd.DataFrame:
    """Apply enrich_listing to every row of a clean-CIAN DataFrame."""
    geocode_cache = _load_json(GEOCODE_CACHE_PATH)
    metro_cache = _load_json(METRO_COORDS_PATH)

    rows: list[dict] = []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        enriched = enrich_listing(
            district=getattr(row, "district", None),
            street=getattr(row, "street", None),
            house_number=getattr(row, "house_number", None),
            underground=getattr(row, "underground", None),
            geocode_cache=geocode_cache,
            metro_cache=metro_cache,
        )
        rows.append(enriched)
        if progress_every and i % progress_every == 0:
            print(f"  geocoded {i}/{n}")

    geo = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), geo], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode a clean CIAN snapshot.")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/cian_spb_clean.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/cian_spb_clean_geo.csv")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Input: {args.input} -> {df.shape}")

    enriched = enrich_dataframe(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)

    n_total = len(enriched)
    n_house = int((enriched["geo_precision"] == "house").sum())
    n_street = int((enriched["geo_precision"] == "street").sum())
    n_district = int((enriched["geo_precision"] == "district").sum())
    n_metro = int(enriched["metro_known"].fillna(False).astype(bool).sum())

    print(f"Output: {args.output} -> {enriched.shape}")
    print(
        f"Coverage: house={n_house} street={n_street} district={n_district} "
        f"metro_known={n_metro}/{n_total}"
    )


if __name__ == "__main__":
    main()
