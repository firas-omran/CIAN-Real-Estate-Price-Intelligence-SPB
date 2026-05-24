"""Generate EDA figures and a compact summary for CIAN data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_eda(df: pd.DataFrame, figures_dir: Path, summary_path: Path) -> None:
    """Create project EDA figures and markdown summary."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["price_mln"] = df["price"] / 1_000_000
    if "target_price_per_sqm" not in df.columns:
        df["target_price_per_sqm"] = df["price"] / df["total_meters"]

    plt.figure(figsize=(9, 5))
    df["price_mln"].hist(bins=40)
    plt.xlabel("Price, mln RUB")
    plt.ylabel("Listings")
    plt.title("Price Distribution")
    save_plot(figures_dir / "price_distribution.png")

    plt.figure(figsize=(9, 5))
    df["log_price"].hist(bins=40)
    plt.xlabel("log1p(price)")
    plt.ylabel("Listings")
    plt.title("Log Price Distribution")
    save_plot(figures_dir / "log_price_distribution.png")

    plt.figure(figsize=(9, 5))
    df["total_meters"].hist(bins=40)
    plt.xlabel("Area, m2")
    plt.ylabel("Listings")
    plt.title("Area Distribution")
    save_plot(figures_dir / "area_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(df["total_meters"], df["price_mln"], alpha=0.45, s=18)
    plt.xlabel("Area, m2")
    plt.ylabel("Price, mln RUB")
    plt.title("Price vs Area")
    save_plot(figures_dir / "price_vs_area.png")

    plt.figure(figsize=(9, 5))
    room_counts = df["rooms_count"].value_counts().sort_index()
    room_counts.plot(kind="bar")
    plt.xlabel("Rooms")
    plt.ylabel("Listings")
    plt.title("Listings by Rooms")
    save_plot(figures_dir / "rooms_distribution.png")

    if "room_segment" in df.columns:
        plt.figure(figsize=(9, 5))
        segment_counts = df["room_segment"].value_counts().sort_index()
        segment_counts.plot(kind="bar")
        plt.xlabel("Room segment")
        plt.ylabel("Listings")
        plt.title("Listings by Collector Room Segment")
        save_plot(figures_dir / "room_segment_distribution.png")

    plt.figure(figsize=(10, 6))
    top_districts = df[df["district"] != "unknown"]["district"].value_counts().head(15).index
    district_prices = df[df["district"].isin(top_districts)].groupby("district")["price_mln"].median().sort_values()
    district_prices.plot(kind="barh")
    plt.xlabel("Median price, mln RUB")
    plt.title("Median Price by District, Top 15 Districts")
    save_plot(figures_dir / "price_by_district_top15.png")

    plt.figure(figsize=(9, 5))
    df.boxplot(column="target_price_per_sqm", by="rooms_count")
    plt.suptitle("")
    plt.title("Price per m2 by Rooms")
    plt.xlabel("Rooms")
    plt.ylabel("RUB per m2")
    save_plot(figures_dir / "price_per_sqm_by_rooms.png")

    plt.figure(figsize=(9, 5))
    missing = df.isna().mean().sort_values(ascending=False).head(20)
    missing.plot(kind="bar")
    plt.ylabel("Missing share")
    plt.title("Missing Values")
    save_plot(figures_dir / "missing_values.png")

    correlation_columns = [
        c
        for c in [
            "price",
            "total_meters",
            "rooms_count",
            "floor",
            "floors_count",
            "target_price_per_sqm",
            "distance_to_center_km",
            "distance_to_metro_km",
        ]
        if c in df.columns
    ]
    numeric = df[correlation_columns].corr()
    plt.figure(figsize=(8, 7))
    plt.imshow(numeric, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(numeric.columns)), numeric.columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric.index)), numeric.index)
    plt.title("Correlation Matrix")
    save_plot(figures_dir / "correlation_matrix.png")

    geo_block = _make_geo_section(df, figures_dir)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    freshness = df["collected_at"].min(), df["collected_at"].max()
    room_segment_distribution = (
        df["room_segment"].value_counts().sort_index().to_string()
        if "room_segment" in df.columns
        else "not available"
    )
    summary = f"""# CIAN SPB EDA Summary

- Rows after cleaning: {len(df)}
- Collection window: {freshness[0]} - {freshness[1]}
- Median price: {df["price"].median():,.0f} RUB
- Mean price: {df["price"].mean():,.0f} RUB
- Median area: {df["total_meters"].median():.1f} m2
- Median price per m2: {df["target_price_per_sqm"].median():,.0f} RUB
- Room distribution:

{df["rooms_count"].value_counts().sort_index().to_string()}

- Room segment distribution:

{room_segment_distribution}

{geo_block}
- Figures directory: `{figures_dir}`
"""
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


def _make_geo_section(df: pd.DataFrame, figures_dir: Path) -> str:
    """Render geocoding coverage block plus geo-specific figures.

    Returns the markdown block to inline into the EDA summary. Skips silently
    if the dataframe was produced before geocoding was wired in.
    """
    geo_columns = {
        "lat",
        "lon",
        "geo_precision",
        "distance_to_center_km",
        "distance_to_metro_km",
        "metro_known",
    }
    if not geo_columns.issubset(df.columns):
        return "## Geocoding\n\n- not available in this snapshot (geocoder not run yet).\n"

    n_total = len(df)
    precision_counts = df["geo_precision"].value_counts(dropna=False)
    n_house = int(precision_counts.get("house", 0))
    n_street = int(precision_counts.get("street", 0))
    n_district = int(precision_counts.get("district", 0))
    metro_mask = df["metro_known"].astype(bool)
    n_metro = int(metro_mask.sum())

    plt.figure(figsize=(9, 5))
    df["distance_to_center_km"].hist(bins=40)
    plt.xlabel("Distance to center, km")
    plt.ylabel("Listings")
    plt.title("Distance to Saint Petersburg Center (Дворцовая)")
    save_plot(figures_dir / "distance_to_center_distribution.png")

    plt.figure(figsize=(9, 5))
    df.loc[metro_mask, "distance_to_metro_km"].hist(bins=40)
    plt.xlabel("Distance to named metro station, km")
    plt.ylabel("Listings")
    plt.title("Distance to Metro (only listings with named station)")
    save_plot(figures_dir / "distance_to_metro_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(
        df["distance_to_center_km"],
        df["target_price_per_sqm"],
        alpha=0.4,
        s=14,
    )
    plt.xlabel("Distance to center, km")
    plt.ylabel("Price per m2, RUB")
    plt.title("Price per m2 vs Distance to Center")
    save_plot(figures_dir / "price_per_sqm_vs_distance.png")

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(
        df["lon"],
        df["lat"],
        c=np.log1p(df["target_price_per_sqm"]),
        cmap="viridis",
        alpha=0.55,
        s=14,
    )
    plt.colorbar(sc, label="log1p(price per m2)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("SPB Listings on Map (color = log1p price/m2)")
    save_plot(figures_dir / "spb_map_price_per_sqm.png")

    median_distance_center = df["distance_to_center_km"].median()
    median_distance_metro = df.loc[metro_mask, "distance_to_metro_km"].median()
    route_block = ""
    route_columns = {"distance_to_metro_route_km", "duration_to_metro_route_min"}
    if route_columns.issubset(df.columns):
        routed_mask = df["distance_to_metro_route_km"].notna()
        n_routed = int(routed_mask.sum())
        if n_routed:
            route_block = (
                f"- Metro walking route available: {n_routed} / {n_total} ({n_routed/n_total:.1%})\n"
                f"- Median walking route distance to metro: "
                f"{df.loc[routed_mask, 'distance_to_metro_route_km'].median():.2f} km\n"
                f"- Median walking route duration to metro: "
                f"{df.loc[routed_mask, 'duration_to_metro_route_min'].median():.1f} min\n"
            )

    return f"""## Geocoding Coverage

- Geocoded rows: {n_total} / {n_total} (100%)
- Precision tiers:
  - house  : {n_house} ({n_house/n_total:.1%})
  - street : {n_street} ({n_street/n_total:.1%})
  - district fallback: {n_district} ({n_district/n_total:.1%})
- Metro distance available: {n_metro} / {n_total} ({n_metro/n_total:.1%})
- Median distance to center: {median_distance_center:.2f} km
- Median distance to metro (when known): {median_distance_metro:.2f} km
{route_block}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA figures for cleaned CIAN data.")
    parser.add_argument(
        "--input",
        default="data/processed/cian_spb_clean_geo.csv",
        help="Path to clean (or geocoded) CIAN dataset.",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("data/processed/figures"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/cian_eda_summary.md"))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fallback = Path("data/processed/cian_spb_clean.csv")
        if fallback.exists():
            input_path = fallback
        else:
            raise FileNotFoundError(f"Neither {args.input} nor {fallback} exist")

    df = pd.read_csv(input_path)
    print(f"Input: {input_path} -> {df.shape}")
    make_eda(df, args.figures_dir, args.summary)


if __name__ == "__main__":
    main()
