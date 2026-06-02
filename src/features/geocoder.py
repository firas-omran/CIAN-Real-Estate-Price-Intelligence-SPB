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
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

try:
    from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    GeocoderServiceError = GeocoderTimedOut = GeocoderUnavailable = Exception
    RateLimiter = None
    Nominatim = None

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
ROUTE_CACHE_PATH = CACHE_DIR / "route_cache.json"
METRO_COORDS_PATH = REFERENCE_DIR / "metro_spb_coords.json"
DISTRICT_CENTROIDS_PATH = REFERENCE_DIR / "spb_district_centroids.json"
OPENROUTESERVICE_URL = "https://api.openrouteservice.org/v2/directions/foot-walking"
OPENROUTESERVICE_ENV = "OPENROUTESERVICE_API_KEY"
OPENROUTESERVICE_DELAY_ENV = "OPENROUTESERVICE_DELAY_SECONDS"
DEFAULT_OPENROUTESERVICE_DELAY_SECONDS = 1.5
DGIS_API_KEY_ENV = "DGIS_API_KEY"
DGIS_SUGGEST_URL = "https://catalog.api.2gis.com/3.0/suggests"
DGIS_GEOCODE_URL = "https://catalog.api.2gis.com/3.0/items/geocode"
DGIS_LOCATION = f"{CENTER_LON},{CENTER_LAT}"

UNKNOWN_VALUES = {"unknown", "", "nan", "none"}
ADDRESS_LABEL_PATTERN = re.compile(r"(?=.*[A-Za-zА-Яа-яЁё])(?=.*\d)")

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


@dataclass(frozen=True)
class AddressSuggestion:
    label: str
    lat: float | None = None
    lon: float | None = None
    source: str = "2gis"


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


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _make_address_key(tier: str, district: str, street: str, house: str) -> str:
    parts = [tier, district.strip().lower(), street.strip().lower(), house.strip().lower()]
    return "|".join(parts)


class _Geolocator:
    """Lazy singleton wrapper over geopy.Nominatim with rate limiting and retries."""

    _instance: Optional["_Geolocator"] = None

    def __init__(self) -> None:
        if Nominatim is None or RateLimiter is None:
            raise RuntimeError(
                "geopy is not installed; live Nominatim lookups are unavailable. "
                "Cached geocodes and district fallback will still be used."
            )
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


def _lookup_live(query: str | dict) -> tuple[float, float] | None:
    """Return a live Nominatim lookup, or None when geopy/network is unavailable."""
    try:
        return _Geolocator.instance().lookup(query)
    except RuntimeError:
        return None


def _parse_dgis_point(point: object) -> tuple[float, float] | None:
    """Return (lat, lon) from a 2GIS point object."""
    if isinstance(point, dict):
        lat = point.get("lat")
        lon = point.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        # 2GIS documents WGS84 coordinates as lon, lat in some fields.
        lon, lat = point[0], point[1]
        return float(lat), float(lon)
    return None


def _dgis_api_key() -> str | None:
    _load_dotenv()
    return os.getenv(DGIS_API_KEY_ENV)


def _dgis_item_label(item: dict) -> str:
    for key in ("full_address_name", "address_name", "full_name"):
        value = item.get(key)
        if value:
            return str(value)
    name = item.get("name")
    address_name = item.get("address_name")
    if name and address_name and str(name) not in str(address_name):
        return f"{address_name}, {name}"
    if name:
        return str(name)
    address = item.get("address")
    if isinstance(address, dict):
        parts = [
            address.get("street") or address.get("street_name"),
            address.get("house") or address.get("house_number") or address.get("number"),
        ]
        joined = ", ".join(str(part).strip() for part in parts if part)
        if joined:
            return joined
        value = address.get("full_name") or address.get("name")
        if value:
            return str(value)
    return ""


def _looks_like_house_address(label: str) -> bool:
    text = str(label or "").strip()
    if not ADDRESS_LABEL_PATTERN.search(text):
        return False
    return not text.replace(" ", "").replace("-", "").isdigit()


def dgis_suggest_addresses(query: str, page_size: int = 8) -> list[AddressSuggestion]:
    """Return 2GIS address suggestions near Saint Petersburg."""
    api_key = _dgis_api_key()
    q = str(query or "").strip()
    if not api_key or len(q) < 3:
        return []

    params = {
        "q": q if CITY.lower() in q.lower() else f"{CITY}, {q}",
        "key": api_key,
        "locale": "ru_RU",
        "suggest_type": "address",
        "fields": "items.point,items.full_address_name,items.address,items.address_name",
        "location": DGIS_LOCATION,
        "page_size": page_size,
    }
    try:
        response = requests.get(DGIS_SUGGEST_URL, params=params, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return []

    items = response.json().get("result", {}).get("items", [])
    suggestions: list[AddressSuggestion] = []
    seen: set[str] = set()
    for item in items:
        label = _dgis_item_label(item)
        if not label or label in seen or not _looks_like_house_address(label):
            continue
        point = _parse_dgis_point(item.get("point"))
        if point is not None and not within_spb_serving_area(point):
            continue
        suggestions.append(
            AddressSuggestion(
                label=label,
                lat=point[0] if point else None,
                lon=point[1] if point else None,
            )
        )
        seen.add(label)
    return suggestions


def dgis_geocode_address(query: str) -> GeocodeResult | None:
    """Resolve an address through 2GIS Geocoder API."""
    api_key = _dgis_api_key()
    q = str(query or "").strip()
    if not api_key or len(q) < 3:
        return None

    params = {
        "q": q if CITY.lower() in q.lower() else f"{CITY}, {q}",
        "key": api_key,
        "locale": "ru_RU",
        "fields": "items.point,items.full_address_name,items.address,items.address_name",
        "location": DGIS_LOCATION,
        "sort": "distance",
        "page_size": 5,
    }
    try:
        response = requests.get(DGIS_GEOCODE_URL, params=params, timeout=7)
        response.raise_for_status()
    except requests.RequestException:
        return None

    items = response.json().get("result", {}).get("items", [])
    for item in items:
        point = _parse_dgis_point(item.get("point"))
        if point is not None and within_spb_serving_area(point):
            return GeocodeResult(point[0], point[1], "house")
    return None


def _make_route_key(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Stable cache key for a pair of coordinates."""
    return f"{lat1:.6f},{lon1:.6f}|{lat2:.6f},{lon2:.6f}|foot-walking"


def _parse_openrouteservice_response(payload: dict) -> tuple[float, float] | None:
    """Return (distance_km, duration_min) from ORS directions response."""
    summary = None
    routes = payload.get("routes") or []
    if routes:
        summary = routes[0].get("summary")
    features = payload.get("features") or []
    if summary is None and features:
        summary = features[0].get("properties", {}).get("summary")
    if not summary:
        return None

    distance_m = summary.get("distance")
    duration_s = summary.get("duration")
    if distance_m is None or duration_s is None:
        return None
    return float(distance_m) / 1000.0, float(duration_s) / 60.0


def route_walking_openrouteservice(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    route_cache: dict | None = None,
) -> tuple[float, float] | None:
    """Calculate pedestrian route distance and duration using openrouteservice.

    Requires `OPENROUTESERVICE_API_KEY`. Results are cached locally so repeated
    pipeline runs do not spend API quota for already-routed pairs.
    """
    _load_dotenv()
    api_key = os.getenv(OPENROUTESERVICE_ENV)
    if not api_key:
        return None

    if route_cache is None:
        route_cache = _load_json(ROUTE_CACHE_PATH)

    key = _make_route_key(origin_lat, origin_lon, dest_lat, dest_lon)
    cached = route_cache.get(key)
    if cached:
        return float(cached["distance_km"]), float(cached["duration_min"])

    delay_seconds = float(os.getenv(OPENROUTESERVICE_DELAY_ENV, DEFAULT_OPENROUTESERVICE_DELAY_SECONDS))
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {"coordinates": [[origin_lon, origin_lat], [dest_lon, dest_lat]]}

    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTESERVICE_URL,
                headers=headers,
                json=body,
                timeout=20,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else delay_seconds * (attempt + 2)
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(delay_seconds * (attempt + 1))
    else:
        return None

    if response is None:
        return None

    parsed = _parse_openrouteservice_response(response.json())
    if parsed is None:
        return None

    distance_km, duration_min = parsed
    route_cache[key] = {"distance_km": distance_km, "duration_min": duration_min}
    _save_json(route_cache, ROUTE_CACHE_PATH)
    return distance_km, duration_min


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

    if house_clean and street_clean:
        key = _make_address_key("house", district_clean, street_clean, house_clean)
        cached = cache.get(key)
        if cached:
            return GeocodeResult(cached["lat"], cached["lon"], "house")
        query = {"street": f"{street_clean} {house_clean}", "city": CITY}
        latlon = _lookup_live(query)
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
        latlon = _lookup_live(query)
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

    query = f"станция метро {name}, {CITY}"
    latlon = _lookup_live(query)
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

    for district in missing:
        query = f"{district} район, {CITY}"
        latlon = _lookup_live(query)
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


def within_spb_serving_area(latlon: tuple[float, float]) -> bool:
    """Stricter serving-time guard for ambiguous user-entered addresses.

    The offline pipeline keeps a broader bbox because CIAN has listings across
    all official SPB districts.  At serving time we use a tighter demo guard so
    vague addresses do not resolve to nearby satellite towns such as Pushkin.
    """
    lat, lon = latlon
    return 59.80 <= lat <= 60.15 and 29.65 <= lon <= 30.70


def geocode_freeform_address(
    street: str | None,
    house_number: str | None,
    cache: dict | None = None,
) -> GeocodeResult | None:
    """Resolve a user-entered SPB address without requiring district input."""
    if cache is None:
        cache = _load_json(GEOCODE_CACHE_PATH)

    street_clean = "" if _is_unknown(street) else str(street).strip()
    house_clean = "" if _is_unknown(house_number) else str(house_number).strip()
    full_address = f"{street_clean} {house_clean}".strip()
    if not full_address:
        return None
    legacy_street = street_clean
    legacy_house = house_clean
    if not legacy_house and "," in street_clean:
        maybe_street, maybe_house = street_clean.rsplit(",", 1)
        legacy_street = maybe_street.strip()
        legacy_house = maybe_house.strip()

    dgis_result = dgis_geocode_address(full_address)
    if dgis_result is not None:
        key = _make_address_key("serving_house", "", full_address, "")
        cache[key] = {"lat": dgis_result.lat, "lon": dgis_result.lon}
        _save_json(cache, GEOCODE_CACHE_PATH)
        return dgis_result

    key = _make_address_key("serving_house", "", full_address, "")
    cached = cache.get(key)
    if cached:
        latlon = (float(cached["lat"]), float(cached["lon"]))
        if within_spb_serving_area(latlon):
            return GeocodeResult(latlon[0], latlon[1], "house")
        return None

    if legacy_street and legacy_house:
        legacy_suffix = f"|{legacy_street.strip().lower()}|{legacy_house.strip().lower()}"
        for legacy_key, legacy_value in cache.items():
            if not legacy_key.startswith("house|") or not legacy_key.endswith(legacy_suffix):
                continue
            latlon = (float(legacy_value["lat"]), float(legacy_value["lon"]))
            if within_spb_serving_area(latlon):
                cache[key] = {"lat": latlon[0], "lon": latlon[1]}
                _save_json(cache, GEOCODE_CACHE_PATH)
                return GeocodeResult(latlon[0], latlon[1], "house")

    return None


def nearest_district(lat: float, lon: float) -> str | None:
    """Infer the closest known SPB district centroid for serving features."""
    centroids = _load_district_centroids_or_seed()
    if not centroids:
        return None
    return min(
        centroids,
        key=lambda district: haversine_km(lat, lon, centroids[district]["lat"], centroids[district]["lon"]),
    )


def nearest_metro(lat: float, lon: float, cache: dict | None = None) -> tuple[str, float] | None:
    """Return closest metro station name and haversine distance in km."""
    if cache is None:
        cache = _load_json(METRO_COORDS_PATH)
    candidates = {
        name: coords
        for name, coords in cache.items()
        if isinstance(coords, dict) and "lat" in coords and "lon" in coords
    }
    if not candidates:
        candidates = METRO_SEED_COORDS
    if not candidates:
        return None

    name = min(
        candidates,
        key=lambda station: haversine_km(lat, lon, candidates[station]["lat"], candidates[station]["lon"]),
    )
    coords = candidates[name]
    return name, haversine_km(lat, lon, float(coords["lat"]), float(coords["lon"]))


def enrich_listing(
    district: str | None,
    street: str | None,
    house_number: str | None,
    underground: str | None,
    geocode_cache: dict | None = None,
    metro_cache: dict | None = None,
    route_cache: dict | None = None,
    with_routing: bool = False,
) -> dict:
    """Compute geo features for a single listing or an API request.

    Returns a dict with keys: lat, lon, geo_precision, distance features,
    and metro_known.
    """
    address_result = geocode_address(district, street, house_number, cache=geocode_cache)
    if address_result is None:
        return {
            "lat": None,
            "lon": None,
            "geo_precision": None,
            "distance_to_center_km": None,
            "distance_to_metro_km": None,
            "distance_to_metro_route_km": None,
            "duration_to_metro_route_min": None,
            "metro_route_provider": None,
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

    route_distance_km = None
    route_duration_min = None
    route_provider = None
    if with_routing and metro_known and metro_latlon is not None:
        routed = route_walking_openrouteservice(
            address_result.lat,
            address_result.lon,
            metro_latlon[0],
            metro_latlon[1],
            route_cache=route_cache,
        )
        if routed is not None:
            route_distance_km, route_duration_min = routed
            route_provider = "openrouteservice:foot-walking"

    return {
        "lat": address_result.lat,
        "lon": address_result.lon,
        "geo_precision": address_result.precision,
        "distance_to_center_km": distance_center,
        "distance_to_metro_km": distance_metro,
        "distance_to_metro_route_km": route_distance_km,
        "duration_to_metro_route_min": route_duration_min,
        "metro_route_provider": route_provider,
        "metro_known": metro_known,
    }


# --- CLI: enrich a clean snapshot ---------------------------------------------


def enrich_dataframe(df: pd.DataFrame, progress_every: int = 50, with_routing: bool = False) -> pd.DataFrame:
    """Apply enrich_listing to every row of a clean-CIAN DataFrame."""
    geocode_cache = _load_json(GEOCODE_CACHE_PATH)
    metro_cache = _load_json(METRO_COORDS_PATH)
    route_cache = _load_json(ROUTE_CACHE_PATH)

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
            route_cache=route_cache,
            with_routing=with_routing,
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
    parser.add_argument(
        "--with-routing",
        action="store_true",
        help=(
            "Call openrouteservice foot-walking API for real walking distance to metro. "
            "Requires OPENROUTESERVICE_API_KEY."
        ),
    )
    args = parser.parse_args()
    _load_dotenv()

    df = pd.read_csv(args.input)
    print(f"Input: {args.input} -> {df.shape}")

    if args.with_routing and not os.getenv(OPENROUTESERVICE_ENV):
        print(
            f"Routing requested, but {OPENROUTESERVICE_ENV} is not set. "
            "Route distance columns will stay empty."
        )

    enriched = enrich_dataframe(df, with_routing=args.with_routing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)

    n_total = len(enriched)
    n_house = int((enriched["geo_precision"] == "house").sum())
    n_street = int((enriched["geo_precision"] == "street").sum())
    n_district = int((enriched["geo_precision"] == "district").sum())
    n_metro = int(enriched["metro_known"].fillna(False).astype(bool).sum())
    n_routed = int(enriched["distance_to_metro_route_km"].notna().sum())

    print(f"Output: {args.output} -> {enriched.shape}")
    print(
        f"Coverage: house={n_house} street={n_street} district={n_district} "
        f"metro_known={n_metro}/{n_total} routed_metro={n_routed}/{n_total}"
    )


if __name__ == "__main__":
    main()
