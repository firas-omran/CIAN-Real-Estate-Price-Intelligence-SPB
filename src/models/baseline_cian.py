"""Non-ML baselines for CIAN apartment price estimation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    name: str
    mae: float
    rmse: float
    mape: float
    mdape: float
    wape: float


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple reproducible split for checkpoint baseline experiments."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled.iloc[:split_idx].copy(), shuffled.iloc[split_idx:].copy()


def metrics(name: str, actual: pd.Series, predicted: pd.Series) -> MetricResult:
    """Calculate business-readable regression metrics."""
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    abs_error = (predicted - actual).abs()
    ape = abs_error / actual
    return MetricResult(
        name=name,
        mae=float(abs_error.mean()),
        rmse=float(np.sqrt(((predicted - actual) ** 2).mean())),
        mape=float(ape.mean() * 100),
        mdape=float(ape.median() * 100),
        wape=float(abs_error.sum() / actual.sum() * 100),
    )


def predict_global_median(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    return pd.Series(train["price"].median(), index=test.index)


def predict_rooms_price_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    global_ppm = train["price_per_sqm_eda"].median()
    ppm_by_rooms = train.groupby("rooms_count")["price_per_sqm_eda"].median()
    ppm = test["rooms_count"].map(ppm_by_rooms).fillna(global_ppm)
    return ppm * test["total_meters"]


def predict_district_rooms_price_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    global_ppm = train["price_per_sqm_eda"].median()
    rooms_ppm = train.groupby("rooms_count")["price_per_sqm_eda"].median()
    district_rooms = train.groupby(["district", "rooms_count"])["price_per_sqm_eda"].median()

    predictions = []
    for _, row in test.iterrows():
        key = (row["district"], row["rooms_count"])
        if key in district_rooms.index:
            ppm = district_rooms.loc[key]
        else:
            ppm = rooms_ppm.get(row["rooms_count"], global_ppm)
        predictions.append(ppm * row["total_meters"])
    return pd.Series(predictions, index=test.index)


def predict_comparable_knn(train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> pd.Series:
    """Comparable listings baseline using only train prices.

    This is a non-ML market baseline: for each test listing, find train listings
    in the same district and room segment, choose closest areas, and multiply
    their median price per m2 by the target area.
    """
    global_ppm = train["price_per_sqm_eda"].median()
    predictions = []

    for _, row in test.iterrows():
        pool = train[
            (train["district"] == row["district"])
            & (train["rooms_count"] == row["rooms_count"])
        ]
        if len(pool) < 3:
            pool = train[train["rooms_count"] == row["rooms_count"]]
        if len(pool) < 3:
            predictions.append(global_ppm * row["total_meters"])
            continue

        neighbors = pool.assign(area_distance=(pool["total_meters"] - row["total_meters"]).abs())
        ppm = neighbors.nsmallest(min(k, len(neighbors)), "area_distance")["price_per_sqm_eda"].median()
        predictions.append(ppm * row["total_meters"])

    return pd.Series(predictions, index=test.index)


def evaluate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    train, test = train_test_split(df)
    actual = test["price"]
    results = [
        metrics("B0 global median price", actual, predict_global_median(train, test)),
        metrics("B1 median price_per_m2 by rooms * area", actual, predict_rooms_price_per_sqm(train, test)),
        metrics("B2 median price_per_m2 by district+rooms * area", actual, predict_district_rooms_price_per_sqm(train, test)),
        metrics("B3 comparable listings KNN baseline", actual, predict_comparable_knn(train, test)),
    ]
    return pd.DataFrame([item.__dict__ for item in results])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CIAN non-ML baselines.")
    parser.add_argument("--input", default="data/processed/cian_spb_clean.csv")
    parser.add_argument("--output", type=Path, default=Path("data/processed/baseline_metrics.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = evaluate_baselines(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
