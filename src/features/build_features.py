"""Build offline and online feature-store artifacts for CIAN data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/cian_spb_clean_geo.csv")
DEFAULT_OUTPUT_DIR = Path("data/features")

BASE_FEATURES = [
    "listing_id",
    "source",
    "collected_at",
    "location_query",
    "author_type",
    "room_segment",
    "rooms_count",
    "total_meters",
    "log_total_meters",
    "floor",
    "floors_count",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "district",
    "underground",
    "residential_complex",
    "lat",
    "lon",
    "geo_precision",
    "distance_to_center_km",
    "distance_to_metro_km",
    "metro_known",
]

TARGET_COLUMNS = ["price", "log_price", "target_price_per_sqm", "log_target_price_per_sqm"]


def build_market_aggregates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build market aggregate tables for offline training and online serving."""
    aggregate_specs = [
        ("district", ("district",)),
        ("district_rooms", ("district", "rooms_count")),
        ("underground", ("underground",)),
        ("room_segment", ("room_segment",)),
        ("rooms", ("rooms_count",)),
    ]

    aggregates: dict[str, pd.DataFrame] = {}
    for table_name, group_cols in aggregate_specs:
        prefix = table_name
        grouped = (
            df.groupby(list(group_cols), dropna=False)
            .agg(
                **{
                    f"{prefix}_ads_count": ("listing_id", "count"),
                    f"{prefix}_median_price": ("price", "median"),
                    f"{prefix}_median_price_per_sqm": ("target_price_per_sqm", "median"),
                    f"{prefix}_p25_price_per_sqm": ("target_price_per_sqm", lambda s: s.quantile(0.25)),
                    f"{prefix}_p75_price_per_sqm": ("target_price_per_sqm", lambda s: s.quantile(0.75)),
                }
            )
            .reset_index()
        )
        aggregates[table_name] = grouped

    return aggregates


def attach_aggregates(df: pd.DataFrame, aggregates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach aggregate feature lookups to listing-level features."""
    featured = df.copy()
    join_plan = {
        "district": ["district"],
        "district_rooms": ["district", "rooms_count"],
        "underground": ["underground"],
        "rooms": ["rooms_count"],
    }

    for table_name, join_cols in join_plan.items():
        featured = featured.merge(aggregates[table_name], on=join_cols, how="left")

    return featured


def add_targets(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Compute training targets: price_per_sqm and its log1p."""
    out = clean_df.copy()
    out["target_price_per_sqm"] = out["price"] / out["total_meters"]
    out["log_target_price_per_sqm"] = np.log1p(out["target_price_per_sqm"])
    return out


def build_offline_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build a model-ready offline feature table and aggregate lookup tables."""
    enriched = add_targets(clean_df)
    aggregates = build_market_aggregates(enriched)

    available_base = [col for col in BASE_FEATURES if col in enriched.columns]
    available_targets = [col for col in TARGET_COLUMNS if col in enriched.columns]
    feature_source = enriched[available_base + available_targets].copy()
    offline = attach_aggregates(feature_source, aggregates)
    return offline, aggregates


def build_feature_registry() -> list[dict[str, object]]:
    """Return a lightweight feature registry for documentation and checks."""
    return [
        {
            "feature": "room_segment",
            "source": "collector segment",
            "type": "categorical",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot / user input",
            "description": "Reliable room segment from collector: studio, 1room, 2rooms, 3rooms, 4rooms.",
            "leakage_risk": "low",
        },
        {
            "feature": "rooms_count",
            "source": "CIAN normalized listing",
            "type": "numeric/categorical",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot",
            "description": "Number of rooms parsed from listing segment.",
            "leakage_risk": "low",
        },
        {
            "feature": "total_meters",
            "source": "CIAN normalized listing",
            "type": "numeric",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Apartment area in square meters; also used to reconstruct price from price_per_sqm prediction.",
            "leakage_risk": "low",
        },
        {
            "feature": "floor_ratio",
            "source": "floor / floors_count",
            "type": "numeric",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Relative floor position in the building.",
            "leakage_risk": "low",
        },
        {
            "feature": "is_first_floor, is_last_floor",
            "source": "floor, floors_count",
            "type": "boolean",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Floor-position flags that often affect apartment liquidity and price.",
            "leakage_risk": "low",
        },
        {
            "feature": "district",
            "source": "CIAN listing",
            "type": "categorical",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot / user input",
            "description": "Saint Petersburg district from VALID_SPB_DISTRICTS whitelist; coarse spatial market segment.",
            "leakage_risk": "low",
        },
        {
            "feature": "underground",
            "source": "CIAN listing",
            "type": "categorical",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot / user input",
            "description": "Nearest metro station name as named in the listing; ~96% of rows have a real station, the rest are 'unknown'.",
            "leakage_risk": "low",
        },
        {
            "feature": "lat, lon",
            "source": "geocoder (Nominatim) of street + house_number + district",
            "type": "numeric",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Geographic coordinates of the listing; resolved with three precision tiers (house/street/district fallback).",
            "leakage_risk": "low",
        },
        {
            "feature": "geo_precision",
            "source": "geocoder result tier",
            "type": "categorical",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Geocoding precision: 'house' / 'street' / 'district'. Lets the model downweight low-precision rows.",
            "leakage_risk": "low",
        },
        {
            "feature": "distance_to_center_km",
            "source": "haversine(lat, lon, Дворцовая 59.9386, 30.3141)",
            "type": "numeric",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Distance from the listing to Saint Petersburg center (Palace Square) in kilometers.",
            "leakage_risk": "low",
        },
        {
            "feature": "distance_to_metro_km",
            "source": "haversine(lat, lon, named_station_lat, named_station_lon)",
            "type": "numeric",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Distance to the metro station named in the listing; NaN when the listing did not name a station.",
            "leakage_risk": "low",
        },
        {
            "feature": "metro_known",
            "source": "underground != 'unknown' and station resolved by geocoder",
            "type": "boolean",
            "offline": True,
            "online": True,
            "refresh": "per request / listing input",
            "description": "Flag indicating whether distance_to_metro_km is meaningful (True) or missing (False).",
            "leakage_risk": "low",
        },
        {
            "feature": "district_rooms_median_price_per_sqm",
            "source": "market aggregate from cleaned CIAN snapshot",
            "type": "numeric aggregate",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot",
            "description": "Median market price per square meter for district and room segment.",
            "leakage_risk": "medium: compute on train split for experiments",
        },
        {
            "feature": "district_rooms_ads_count",
            "source": "market aggregate from cleaned CIAN snapshot",
            "type": "numeric aggregate",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot",
            "description": "Number of comparable listings in district and room segment.",
            "leakage_risk": "low",
        },
        {
            "feature": "underground_median_price_per_sqm",
            "source": "market aggregate from cleaned CIAN snapshot",
            "type": "numeric aggregate",
            "offline": True,
            "online": True,
            "refresh": "weekly snapshot",
            "description": "Median market price per square meter near metro station.",
            "leakage_risk": "medium: compute on train split for experiments",
        },
        {
            "feature": "target_price_per_sqm, log_target_price_per_sqm",
            "source": "price / total_meters and its log1p",
            "type": "target",
            "offline": True,
            "online": False,
            "refresh": "weekly snapshot",
            "description": "Active modeling target. Models are trained on log_target_price_per_sqm; price is reconstructed as expm1(pred) * total_meters for business metrics.",
            "leakage_risk": "target, never use as input feature",
        },
        {
            "feature": "price, log_price",
            "source": "CIAN listing",
            "type": "target",
            "offline": True,
            "online": False,
            "refresh": "weekly snapshot",
            "description": "Original raw price target; kept for backward compatibility with checkpoint 1 baselines and business metrics.",
            "leakage_risk": "target, never use as input feature",
        },
    ]


def write_feature_registry(registry: list[dict[str, object]], output_dir: Path) -> None:
    """Save registry as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feature_registry.json"
    md_path = Path("docs/feature_registry.md")

    json_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        "| Feature | Source | Offline | Online | Refresh | Leakage risk |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in registry:
        rows.append(
            "| {feature} | {source} | {offline} | {online} | {refresh} | {leakage_risk} |".format(
                **item
            )
        )
    md_path.write_text(
        "# Feature Registry\n\n"
        "This registry documents the current feature set for Checkpoint 2 and separates "
        "offline training features from online serving features.\n\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CIAN feature-store artifacts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    clean_df = pd.read_csv(args.input)
    offline, aggregates = build_offline_features(clean_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    offline_path = args.output_dir / "cian_spb_offline_features.csv"
    offline.to_csv(offline_path, index=False)

    for name, table in aggregates.items():
        table.to_csv(args.output_dir / f"cian_spb_{name}_market_aggregates.csv", index=False)

    write_feature_registry(build_feature_registry(), args.output_dir)

    print(f"Input clean data: {args.input} -> {clean_df.shape}")
    print(f"Offline feature table: {offline_path} -> {offline.shape}")
    print("Aggregate tables:")
    for name, table in aggregates.items():
        print(f"  {name}: {table.shape}")


if __name__ == "__main__":
    main()
