"""Generate EDA figures and a compact summary for CIAN data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_eda(df: pd.DataFrame, figures_dir: Path, summary_path: Path) -> None:
    """Create project EDA figures and markdown summary."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    df["price_mln"] = df["price"] / 1_000_000

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

    plt.figure(figsize=(10, 6))
    top_districts = df[df["district"] != "unknown"]["district"].value_counts().head(15).index
    district_prices = df[df["district"].isin(top_districts)].groupby("district")["price_mln"].median().sort_values()
    district_prices.plot(kind="barh")
    plt.xlabel("Median price, mln RUB")
    plt.title("Median Price by District, Top 15 Districts")
    save_plot(figures_dir / "price_by_district_top15.png")

    plt.figure(figsize=(9, 5))
    df.boxplot(column="price_per_sqm_eda", by="rooms_count")
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

    numeric = df[["price", "total_meters", "rooms_count", "floor", "floors_count", "price_per_sqm_eda"]].corr()
    plt.figure(figsize=(7, 6))
    plt.imshow(numeric, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(numeric.columns)), numeric.columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric.index)), numeric.index)
    plt.title("Correlation Matrix")
    save_plot(figures_dir / "correlation_matrix.png")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    freshness = df["collected_at"].min(), df["collected_at"].max()
    summary = f"""# CIAN SPB EDA Summary

- Rows after cleaning: {len(df)}
- Collection window: {freshness[0]} - {freshness[1]}
- Median price: {df["price"].median():,.0f} RUB
- Mean price: {df["price"].mean():,.0f} RUB
- Median area: {df["total_meters"].median():.1f} m2
- Median price per m2: {df["price_per_sqm_eda"].median():,.0f} RUB
- Room distribution:

{df["rooms_count"].value_counts().sort_index().to_string()}

- Figures directory: `{figures_dir}`
"""
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA figures for cleaned CIAN data.")
    parser.add_argument("--input", default="data/processed/cian_spb_clean.csv")
    parser.add_argument("--figures-dir", type=Path, default=Path("data/processed/figures"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/cian_eda_summary.md"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    make_eda(df, args.figures_dir, args.summary)


if __name__ == "__main__":
    main()
