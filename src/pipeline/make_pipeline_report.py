"""Create a compact Markdown report for the current data pipeline run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def shape_line(path: Path) -> str:
    """Return a markdown table row with CSV shape."""
    if not path.exists():
        return f"| `{path}` | missing | missing |"
    df = pd.read_csv(path)
    return f"| `{path}` | {len(df)} | {len(df.columns)} |"


def latest_normalized_path() -> Path | None:
    files = sorted(Path("data/raw").glob("cian_spb_normalized_*.csv"))
    return files[-1] if files else None


def make_report(output: Path) -> None:
    normalized = latest_normalized_path()
    clean = Path("data/processed/cian_spb_clean.csv")
    geocoded = Path("data/processed/cian_spb_clean_geo.csv")
    offline = Path("data/features/cian_spb_offline_features.csv")
    balanced = Path("data/processed/cian_spb_balanced_sample.csv")

    clean_df = pd.read_csv(clean) if clean.exists() else pd.DataFrame()
    geocoded_df = pd.read_csv(geocoded) if geocoded.exists() else pd.DataFrame()
    balanced_df = pd.read_csv(balanced) if balanced.exists() else pd.DataFrame()

    artifact_rows = [
        "| Artifact | Rows | Columns |",
        "|---|---:|---:|",
    ]
    if normalized:
        artifact_rows.append(shape_line(normalized))
    artifact_rows.extend([
        shape_line(clean),
        shape_line(geocoded),
        shape_line(offline),
        shape_line(Path("data/features/cian_spb_district_market_aggregates.csv")),
        shape_line(Path("data/features/cian_spb_district_rooms_market_aggregates.csv")),
        shape_line(Path("data/features/cian_spb_underground_market_aggregates.csv")),
        shape_line(Path("data/features/cian_spb_room_segment_market_aggregates.csv")),
        shape_line(Path("data/features/cian_spb_rooms_market_aggregates.csv")),
        shape_line(balanced),
    ])

    room_distribution = (
        clean_df["rooms_count"].value_counts().sort_index().to_string()
        if "rooms_count" in clean_df.columns
        else "not available"
    )
    segment_distribution = (
        clean_df["room_segment"].value_counts().sort_index().to_string()
        if "room_segment" in clean_df.columns
        else "not available"
    )
    balanced_distribution = (
        balanced_df["rooms_count"].value_counts().sort_index().to_string()
        if "rooms_count" in balanced_df.columns
        else "not available"
    )
    route_summary = "not available"
    if {"distance_to_metro_route_km", "duration_to_metro_route_min"}.issubset(geocoded_df.columns):
        routed = geocoded_df["distance_to_metro_route_km"].notna()
        route_summary = (
            f"- Metro route distance coverage: {int(routed.sum())} / {len(geocoded_df)}\n"
            f"- Median route distance to metro: "
            f"{geocoded_df.loc[routed, 'distance_to_metro_route_km'].median():.2f} km\n"
            f"- Median route duration to metro: "
            f"{geocoded_df.loc[routed, 'duration_to_metro_route_min'].median():.1f} min"
        )

    artifacts_table = "\n".join(artifact_rows)

    report = f"""# Data Pipeline Report

## Artifacts

{artifacts_table}

## Clean Data Summary

- Data Contract status: OK if `python -m src.data.contract_cian data/processed/cian_spb_clean.csv` passes.
- Clean rows: {len(clean_df)}
- Median price: {clean_df["price"].median():,.0f} RUB
- Median area: {clean_df["total_meters"].median():.1f} m2

Room distribution:

{room_distribution}

Room segment distribution:

{segment_distribution}

## Metro Routing

{route_summary}

## Sampling

Balanced sample distribution:

{balanced_distribution}

## Feature Store

- Offline features: `data/features/cian_spb_offline_features.csv`
- Online lookup tables: district, district+rooms, underground, room_segment, rooms.
- Registry: `docs/feature_registry.md` and `data/features/feature_registry.json`.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pipeline report.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/pipeline_report.md"))
    args = parser.parse_args()
    make_report(args.output)


if __name__ == "__main__":
    main()
