"""Non-ML baselines for CIAN apartment price estimation on the price_per_sqm target.

Each baseline B0..B5 predicts price per square meter directly. The final
price prediction is reconstructed as `price_per_sqm_pred * total_meters` so
business metrics stay comparable with the original ML System Design Doc.

B0-B3 are non-spatial baselines on price_per_sqm.
B4-B5 incorporate continuous geo features (distance to center / metro).

Reported metrics:
    * MAE  — mean absolute error of reconstructed price, RUB
    * MAPE — mean absolute percentage error of reconstructed price, %
    * R²   — coefficient of determination on reconstructed price
    * R²_per_sqm — coefficient of determination on price_per_sqm itself.
        This is the *honest* metric: it shows whether the baseline understands
        the local market beyond mechanical scaling by area.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PER_SQM_COLUMN = "price_per_sqm"

CENTER_BINS = [0, 3, 6, 10, 15, float("inf")]
CENTER_LABELS = ["0-3km", "3-6km", "6-10km", "10-15km", "15km+"]

METRO_BINS = [0, 0.5, 1, 2, 5, float("inf")]
METRO_LABELS = ["0-0.5km", "0.5-1km", "1-2km", "2-5km", "5km+"]


@dataclass
class MetricResult:
    name: str
    mae: float
    mape: float
    r2_price: float
    r2_per_sqm: float


def _r2(actual: pd.Series, predicted: pd.Series) -> float:
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    ss_res = float(((actual - predicted) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def metrics(
    name: str,
    actual_price: pd.Series,
    predicted_price: pd.Series,
    actual_per_sqm: pd.Series,
    predicted_per_sqm: pd.Series,
) -> MetricResult:
    """Calculate business-readable metrics on price and the honest R² on per_sqm."""
    abs_error = (predicted_price - actual_price).abs()
    ape = abs_error / actual_price
    return MetricResult(
        name=name,
        mae=float(abs_error.mean()),
        mape=float(ape.mean() * 100),
        r2_price=_r2(actual_price, predicted_price),
        r2_per_sqm=_r2(actual_per_sqm, predicted_per_sqm),
    )


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible split that matches the historical baseline split for comparability."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled.iloc[:split_idx].copy(), shuffled.iloc[split_idx:].copy()


def predict_global_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    return pd.Series(train[PER_SQM_COLUMN].median(), index=test.index)


def predict_rooms_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    global_ppm = train[PER_SQM_COLUMN].median()
    ppm_by_rooms = train.groupby("rooms_count")[PER_SQM_COLUMN].median()
    return test["rooms_count"].map(ppm_by_rooms).fillna(global_ppm)


def predict_district_rooms_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    global_ppm = train[PER_SQM_COLUMN].median()
    rooms_ppm = train.groupby("rooms_count")[PER_SQM_COLUMN].median()
    district_rooms = train.groupby(["district", "rooms_count"])[PER_SQM_COLUMN].median()

    predictions = []
    for _, row in test.iterrows():
        key = (row["district"], row["rooms_count"])
        if key in district_rooms.index:
            ppm = district_rooms.loc[key]
        else:
            ppm = rooms_ppm.get(row["rooms_count"], global_ppm)
        predictions.append(ppm)
    return pd.Series(predictions, index=test.index)


def predict_comparable_knn_per_sqm(train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> pd.Series:
    """Comparable listings baseline: same district + rooms, nearest by area, median per_sqm."""
    global_ppm = train[PER_SQM_COLUMN].median()
    predictions = []

    for _, row in test.iterrows():
        pool = train[
            (train["district"] == row["district"])
            & (train["rooms_count"] == row["rooms_count"])
        ]
        if len(pool) < 3:
            pool = train[train["rooms_count"] == row["rooms_count"]]
        if len(pool) < 3:
            predictions.append(global_ppm)
            continue

        neighbors = pool.assign(area_distance=(pool["total_meters"] - row["total_meters"]).abs())
        ppm = neighbors.nsmallest(min(k, len(neighbors)), "area_distance")[PER_SQM_COLUMN].median()
        predictions.append(ppm)

    return pd.Series(predictions, index=test.index)


def _assign_center_bin(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["distance_to_center_km"], bins=CENTER_BINS, labels=CENTER_LABELS, right=False)


def _assign_metro_bin(df: pd.DataFrame) -> pd.Series:
    metro_bin = pd.cut(df["distance_to_metro_km"], bins=METRO_BINS, labels=METRO_LABELS, right=False)
    metro_bin = metro_bin.astype(object)
    metro_bin[~df["metro_known"].astype(bool)] = "unknown"
    return metro_bin


def predict_center_bins_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """B4: median price_per_sqm by (center_distance_bin, rooms_count)."""
    train = train.copy()
    train["center_bin"] = _assign_center_bin(train)
    global_ppm = train[PER_SQM_COLUMN].median()
    rooms_ppm = train.groupby("rooms_count")[PER_SQM_COLUMN].median()
    bin_rooms = train.groupby(["center_bin", "rooms_count"], observed=False)[PER_SQM_COLUMN].median()

    test = test.copy()
    test["center_bin"] = _assign_center_bin(test)

    predictions = []
    for _, row in test.iterrows():
        key = (row["center_bin"], row["rooms_count"])
        if key in bin_rooms.index:
            predictions.append(bin_rooms.loc[key])
        else:
            predictions.append(rooms_ppm.get(row["rooms_count"], global_ppm))

    return pd.Series(predictions, index=test.index)


def predict_metro_bins_per_sqm(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """B5: median price_per_sqm by (metro_distance_bin, rooms_count)."""
    train = train.copy()
    train["metro_bin"] = _assign_metro_bin(train)
    global_ppm = train[PER_SQM_COLUMN].median()
    rooms_ppm = train.groupby("rooms_count")[PER_SQM_COLUMN].median()
    bin_rooms = train.groupby(["metro_bin", "rooms_count"], observed=False)[PER_SQM_COLUMN].median()

    test = test.copy()
    test["metro_bin"] = _assign_metro_bin(test)

    predictions = []
    for _, row in test.iterrows():
        key = (row["metro_bin"], row["rooms_count"])
        if key in bin_rooms.index:
            predictions.append(bin_rooms.loc[key])
        else:
            predictions.append(rooms_ppm.get(row["rooms_count"], global_ppm))

    return pd.Series(predictions, index=test.index)


def evaluate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate B0..B5 on price_per_sqm and report metrics on reconstructed price."""
    df = df.copy()
    df[PER_SQM_COLUMN] = df["price"] / df["total_meters"]

    train, test = train_test_split(df)
    actual_price = test["price"]
    actual_per_sqm = test[PER_SQM_COLUMN]
    total_meters = test["total_meters"]

    def reconstruct(per_sqm_pred: pd.Series) -> pd.Series:
        return per_sqm_pred * total_meters

    baselines = [
        ("B0 global median price_per_sqm", predict_global_per_sqm),
        ("B1 median price_per_sqm by rooms", predict_rooms_per_sqm),
        ("B2 median price_per_sqm by district + rooms", predict_district_rooms_per_sqm),
        ("B3 comparable listings KNN on price_per_sqm", predict_comparable_knn_per_sqm),
        ("B4 median price_per_sqm by center_distance_bin + rooms", predict_center_bins_per_sqm),
        ("B5 median price_per_sqm by metro_distance_bin + rooms", predict_metro_bins_per_sqm),
    ]

    results = []
    for name, predict_fn in baselines:
        per_sqm_pred = predict_fn(train, test)
        price_pred = reconstruct(per_sqm_pred)
        results.append(
            metrics(name, actual_price, price_pred, actual_per_sqm, per_sqm_pred)
        )
    return pd.DataFrame([item.__dict__ for item in results])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CIAN non-ML baselines on price_per_sqm target.")
    parser.add_argument("--input", default="data/processed/cian_spb_clean_geo.csv")
    parser.add_argument("--output", type=Path, default=Path("data/processed/baseline_metrics.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = evaluate_baselines(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
