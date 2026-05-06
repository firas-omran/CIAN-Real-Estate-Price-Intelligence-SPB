"""Clean normalized CIAN listings for modeling and EDA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATTERN = "cian_spb_normalized_*.csv"
DEFAULT_OUTPUT = Path("data/processed/cian_spb_clean.csv")

VALID_SPB_DISTRICTS: frozenset[str] = frozenset(
    {
        "Адмиралтейский",
        "Василеостровский",
        "Выборгский",
        "Калининский",
        "Кировский",
        "Колпинский",
        "Красногвардейский",
        "Красносельский",
        "Кронштадтский",
        "Курортный",
        "Московский",
        "Невский",
        "Петроградский",
        "Петродворцовый",
        "Приморский",
        "Пушкинский",
        "Фрунзенский",
        "Центральный",
    }
)


def find_latest_input(input_path: str | None) -> Path:
    """Return explicit input path or latest normalized CIAN snapshot."""
    if input_path:
        return Path(input_path)

    files = sorted(Path("data/raw").glob(RAW_PATTERN))
    if not files:
        raise FileNotFoundError(
            "No CIAN normalized files found. Run: python -m src.data.collect_cian_spb --pages 10"
        )
    return files[-1]


def clean_cian_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply project cleaning rules to normalized CIAN data."""
    cleaned = df.copy()

    numeric_columns = [
        "rooms_count",
        "total_meters",
        "price",
        "observed_price_per_sqm",
        "floor",
        "floors_count",
        "floor_ratio",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in ["floor", "floors_count", "floor_ratio"]:
        if column in cleaned.columns:
            cleaned.loc[cleaned[column] < 0, column] = np.nan

    if "listing_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["listing_id"])
    elif "url" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["url"])

    if "room_segment" in cleaned.columns:
        cleaned["room_segment"] = cleaned["room_segment"].fillna("unknown").astype(str).str.strip()
        cleaned.loc[cleaned["room_segment"].eq("studio"), "rooms_count"] = 0

    cleaned = cleaned.dropna(subset=["price", "total_meters", "rooms_count"])
    cleaned = cleaned[cleaned["price"].between(1_000_000, 600_000_000)]
    cleaned = cleaned[cleaned["total_meters"].between(10, 500)]
    cleaned = cleaned[cleaned["rooms_count"].between(0, 10)]

    cleaned["price_per_sqm_eda"] = cleaned["price"] / cleaned["total_meters"]
    cleaned = cleaned[cleaned["price_per_sqm_eda"].between(50_000, 3_000_000)]
    cleaned["log_price"] = np.log1p(cleaned["price"])
    cleaned["log_total_meters"] = np.log1p(cleaned["total_meters"])

    if {"floor", "floors_count"}.issubset(cleaned.columns):
        cleaned["floor_ratio"] = cleaned["floor"] / cleaned["floors_count"]
        cleaned.loc[cleaned["floors_count"].isna() | (cleaned["floors_count"] <= 0), "floor_ratio"] = np.nan
        cleaned["is_first_floor"] = cleaned["floor"].eq(1)
        cleaned["is_last_floor"] = cleaned["floor"].eq(cleaned["floors_count"])

    text_columns = ["district", "street", "underground", "residential_complex", "author_type", "room_segment"]
    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("unknown").astype(str).str.strip()

    if "district" in cleaned.columns:
        cleaned = cleaned[cleaned["district"].isin(VALID_SPB_DISTRICTS)]

    return cleaned.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean normalized CIAN data.")
    parser.add_argument("--input", default=None, help="Path to normalized CIAN CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Clean CSV output path.")
    args = parser.parse_args()

    input_path = find_latest_input(args.input)
    df = pd.read_csv(input_path)
    cleaned = clean_cian_frame(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)

    print(f"Input: {input_path} -> {df.shape}")
    print(f"Output: {args.output} -> {cleaned.shape}")
    print("Removed rows:", len(df) - len(cleaned))
    print(cleaned[["price", "total_meters", "rooms_count", "price_per_sqm_eda"]].describe())


if __name__ == "__main__":
    main()
