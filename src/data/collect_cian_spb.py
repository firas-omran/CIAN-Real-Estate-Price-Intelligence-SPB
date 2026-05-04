"""Collect a diverse Saint Petersburg CIAN sample for the ML project.

The script collects several room segments separately, merges them, removes
duplicate listing URLs, and saves both raw and normalized CSV snapshots.

Example:
    python -m src.data.collect_cian_spb --pages 20 --timeout 20
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.data.collect_cian import collect_flats


ROOM_SEGMENTS: tuple[tuple[str, tuple[int | str, ...]], ...] = (
    ("studio", ("studio",)),
    ("1room", (1,)),
    ("2rooms", (2,)),
    ("3rooms", (3,)),
    ("4rooms", (4,)),
)

NORMALIZED_COLUMNS = [
    "listing_id",
    "url",
    "source",
    "collected_at",
    "location",
    "location_query",
    "deal_type",
    "accommodation_type",
    "author_type",
    "rooms_count",
    "total_meters",
    "price",
    "observed_price_per_sqm",
    "floor",
    "floors_count",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "district",
    "street",
    "house_number",
    "underground",
    "residential_complex",
]


def extract_listing_id(url: object) -> str | None:
    """Extract numeric CIAN listing id from a URL."""
    if not isinstance(url, str):
        return None
    path = urlparse(url).path
    parts = [part for part in path.rstrip("/").split("/") if part]
    return parts[-1] if parts and parts[-1].isdigit() else None


def normalize_cian_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable project schema from raw cianparser output."""
    normalized = df.copy()

    if "url" in normalized.columns:
        normalized["listing_id"] = normalized["url"].map(extract_listing_id)
    else:
        normalized["listing_id"] = None

    numeric_columns = ["rooms_count", "total_meters", "price", "floor", "floors_count"]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if {"price", "total_meters"}.issubset(normalized.columns):
        normalized["observed_price_per_sqm"] = (
            normalized["price"] / normalized["total_meters"]
        ).where(normalized["total_meters"] > 0)
    else:
        normalized["observed_price_per_sqm"] = pd.NA

    if {"floor", "floors_count"}.issubset(normalized.columns):
        normalized["floor_ratio"] = (
            normalized["floor"] / normalized["floors_count"]
        ).where(normalized["floors_count"] > 0)
        normalized["is_first_floor"] = normalized["floor"].eq(1)
        normalized["is_last_floor"] = normalized["floor"].eq(normalized["floors_count"])
    else:
        normalized["floor_ratio"] = pd.NA
        normalized["is_first_floor"] = pd.NA
        normalized["is_last_floor"] = pd.NA

    for column in NORMALIZED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized[NORMALIZED_COLUMNS]
    normalized = normalized.dropna(subset=["price", "total_meters"])
    normalized = normalized[normalized["price"] > 0]
    normalized = normalized[normalized["total_meters"] > 0]
    return normalized


def collect_spb_dataset(
    pages: int,
    output_dir: Path,
    timeout: int,
    with_extra_data: bool,
    proxy: str | None,
) -> tuple[Path, Path]:
    """Collect and save a multi-segment Saint Petersburg dataset."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"cian_spb_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    segment_frames = []
    for segment_name, rooms in ROOM_SEGMENTS:
        print(f"\n=== Collecting {segment_name}: rooms={rooms}, pages=1..{pages} ===")
        segment_path = collect_flats(
            location="Санкт-Петербург",
            start_page=1,
            end_page=pages,
            rooms=rooms,
            output_dir=run_dir,
            with_extra_data=with_extra_data,
            timeout=timeout,
            proxy=proxy,
        )
        frame = pd.read_csv(segment_path)
        frame["room_segment"] = segment_name
        segment_frames.append(frame)

    raw = pd.concat(segment_frames, ignore_index=True, sort=False)
    if "url" in raw.columns:
        raw = raw.drop_duplicates(subset=["url"])

    raw_path = output_dir / f"cian_spb_raw_{timestamp}.csv"
    normalized_path = output_dir / f"cian_spb_normalized_{timestamp}.csv"

    normalized = normalize_cian_frame(raw)
    raw.to_csv(raw_path, index=False)
    normalized.to_csv(normalized_path, index=False)

    print("\nDone.")
    print(f"Raw rows: {len(raw)} -> {raw_path}")
    print(f"Normalized rows: {len(normalized)} -> {normalized_path}")
    print("Normalized columns:")
    print(", ".join(normalized.columns))

    return raw_path, normalized_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a diverse CIAN sale dataset for Saint Petersburg."
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Number of result pages per room segment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where CSV snapshots will be saved.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--with-extra-data",
        action="store_true",
        help="Collect extra details. Slower, but may add useful fields.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optional HTTPS proxy, e.g. http://user:pass@host:port.",
    )
    args = parser.parse_args()

    collect_spb_dataset(
        pages=args.pages,
        output_dir=args.output_dir,
        timeout=args.timeout,
        with_extra_data=args.with_extra_data,
        proxy=args.proxy,
    )


if __name__ == "__main__":
    main()
