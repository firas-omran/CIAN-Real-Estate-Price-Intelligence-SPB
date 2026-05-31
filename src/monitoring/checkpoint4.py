"""Checkpoint 4 monitoring, alerting, and drift-demo utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.api.predictor import load_best_model

FEATURES_PATH = Path("data/features/cian_spb_offline_features.csv")
METRICS_PATH = Path("data/experiments/checkpoint3_metrics.csv")

NUMERIC_DRIFT_COLUMNS = [
    "total_meters",
    "rooms_count",
    "floor_ratio",
    "distance_to_center_km",
    "distance_to_metro_route_km",
    "target_price_per_sqm",
]
CATEGORICAL_DRIFT_COLUMNS = ["district", "rooms_count", "author_type", "geo_precision"]
MODEL_FEATURE_COLUMNS = [
    "rooms_count",
    "total_meters",
    "log_total_meters",
    "floor",
    "floors_count",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "lat",
    "lon",
    "distance_to_center_km",
    "distance_to_metro_route_km",
    "duration_to_metro_route_min",
    "metro_known",
    "author_type",
    "room_segment",
    "district",
    "underground",
    "geo_precision",
    "metro_route_provider",
]


@dataclass
class Alert:
    level: str
    name: str
    condition: str
    cause: str
    action: str
    value: Any
    threshold: Any
    fired: bool


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def make_drifted_snapshot(reference: pd.DataFrame, fraction: float = 0.60) -> pd.DataFrame:
    """Create a deterministic degraded snapshot for the classroom demo."""
    current = reference.copy(deep=True)
    sample = current.sample(frac=fraction, random_state=42).index
    premium_districts = ["Центральный", "Петроградский", "Василеостровский", "Адмиралтейский"]

    current.loc[sample, "total_meters"] = current.loc[sample, "total_meters"] * 2.20
    current.loc[sample, "log_total_meters"] = np.log1p(current.loc[sample, "total_meters"])
    current.loc[sample, "target_price_per_sqm"] = current.loc[sample, "target_price_per_sqm"] * 1.65
    current.loc[sample, "log_target_price_per_sqm"] = np.log1p(current.loc[sample, "target_price_per_sqm"])
    current.loc[sample, "price"] = current.loc[sample, "target_price_per_sqm"] * current.loc[sample, "total_meters"]
    current.loc[sample, "district"] = [premium_districts[i % len(premium_districts)] for i in range(len(sample))]
    current.loc[sample, "geo_precision"] = "district"

    missing_geo = current.sample(frac=0.08, random_state=7).index
    current.loc[missing_geo, ["lat", "lon", "distance_to_center_km"]] = np.nan

    if "collected_at" in current.columns:
        current["collected_at"] = "2026-04-01T00:00:00+00:00"
    return current


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for numeric features."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(cur, edges, include_lowest=True).value_counts(sort=False)
    ref_share = ref_counts / max(ref_counts.sum(), 1)
    cur_share = cur_counts / max(cur_counts.sum(), 1)

    epsilon = 1e-6
    ref_share = ref_share.replace(0, epsilon)
    cur_share = cur_share.replace(0, epsilon)
    return float(((cur_share - ref_share) * np.log(cur_share / ref_share)).sum())


def max_category_share_delta(reference: pd.Series, current: pd.Series) -> float:
    ref_share = reference.astype(str).value_counts(normalize=True).to_dict()
    cur_share = current.astype(str).value_counts(normalize=True).to_dict()
    keys = set(ref_share) | set(cur_share)
    if not keys:
        return 0.0
    return float(max(abs(cur_share.get(key, 0.0) - ref_share.get(key, 0.0)) for key in keys))


def data_quality(current: pd.DataFrame) -> dict[str, Any]:
    newest = pd.to_datetime(current.get("collected_at"), errors="coerce", utc=True).max()
    if pd.isna(newest):
        snapshot_age_days = 10**9
    else:
        snapshot_age_days = (datetime.now(timezone.utc) - newest.to_pydatetime()).total_seconds() / 86400

    geocode_coverage = (
        float(current[["lat", "lon"]].notna().all(axis=1).mean())
        if {"lat", "lon"}.issubset(current.columns)
        else 0.0
    )
    target_missing = float(current["target_price_per_sqm"].isna().mean()) if "target_price_per_sqm" in current else 1.0
    return {
        "rows": int(len(current)),
        "snapshot_age_days": round(snapshot_age_days, 2),
        "geocode_coverage": round(geocode_coverage, 4),
        "target_missing_share": round(target_missing, 4),
    }


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in NUMERIC_DRIFT_COLUMNS:
        if column in reference.columns and column in current.columns:
            rows.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "metric": "psi",
                    "value": psi(reference[column], current[column]),
                    "threshold": 0.20,
                    "drift_detected": psi(reference[column], current[column]) > 0.20,
                }
            )
    for column in CATEGORICAL_DRIFT_COLUMNS:
        if column in reference.columns and column in current.columns:
            value = max_category_share_delta(reference[column], current[column])
            rows.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "metric": "max_share_delta",
                    "value": value,
                    "threshold": 0.10,
                    "drift_detected": value > 0.10,
                }
            )
    return pd.DataFrame(rows).sort_values("value", ascending=False)


def prediction_summary(current: pd.DataFrame) -> dict[str, Any]:
    model_name, model = load_best_model()
    available = [column for column in MODEL_FEATURE_COLUMNS if column in current.columns]
    preds_log = model.predict(current[available])
    pred_per_sqm = np.expm1(preds_log)
    pred_price = pred_per_sqm * current["total_meters"].to_numpy()
    return {
        "model_name": model_name,
        "prediction_count": int(len(pred_price)),
        "predicted_price_median": float(np.median(pred_price)),
        "predicted_price_p95": float(np.percentile(pred_price, 95)),
        "predicted_per_sqm_median": float(np.median(pred_per_sqm)),
        "high_price_share": float((pred_price > 100_000_000).mean()),
    }


def build_alerts(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    system_metrics: dict[str, float] | None = None,
) -> list[Alert]:
    system_metrics = system_metrics or {"p95_latency_ms": 180.0, "error_rate": 0.0}
    quality = data_quality(current)
    drift = drift_report(reference, current)
    pred = prediction_summary(current)

    max_psi = float(drift.loc[drift["metric"] == "psi", "value"].max()) if not drift.empty else 0.0
    max_cat_delta = (
        float(drift.loc[drift["metric"] == "max_share_delta", "value"].max()) if not drift.empty else 0.0
    )

    return [
        Alert(
            "data",
            "snapshot_freshness",
            "snapshot_age_days > 7",
            "CIAN snapshot was not refreshed on schedule.",
            "Run pipeline with --collect, validate contract, then rebuild features.",
            quality["snapshot_age_days"],
            7,
            quality["snapshot_age_days"] > 7,
        ),
        Alert(
            "data",
            "geocode_coverage_drop",
            "geocode_coverage < 0.95",
            "Nominatim/cache/SSL issue or too many incomplete addresses.",
            "Check geocoder logs, certificates, cache, then rerun geocoding.",
            quality["geocode_coverage"],
            0.95,
            quality["geocode_coverage"] < 0.95,
        ),
        Alert(
            "data",
            "numeric_data_drift",
            "max numeric PSI > 0.20",
            "Input distribution changed: area, prices, or location mix shifted.",
            "Inspect drift table; if business shift is real, retrain and compare metrics.",
            round(max_psi, 4),
            0.20,
            max_psi > 0.20,
        ),
        Alert(
            "data",
            "segment_drift",
            "max category share delta > 10 pp",
            "District, room, seller, or geo-precision mix changed.",
            "Check parser/collection query; rebalance snapshot or retrain.",
            round(max_cat_delta, 4),
            0.10,
            max_cat_delta > 0.10,
        ),
        Alert(
            "model",
            "prediction_tail_growth",
            "share(predicted_price > 100M RUB) > 5%",
            "Model is seeing too many luxury/outlier-like objects.",
            "Review high-price requests, cap demo input ranges, retrain with fresh luxury coverage.",
            round(pred["high_price_share"], 4),
            0.05,
            pred["high_price_share"] > 0.05,
        ),
        Alert(
            "system",
            "api_latency_p95",
            "p95 latency > 500 ms",
            "Cold start, overloaded host, or slow geocoding path.",
            "Check logs; restart service; disable live geocoding; rollback if needed.",
            system_metrics["p95_latency_ms"],
            500,
            system_metrics["p95_latency_ms"] > 500,
        ),
        Alert(
            "system",
            "api_error_rate",
            "error_rate > 5%",
            "Bad release, missing model artifact, or invalid inputs.",
            "Rollback to previous image/model; inspect stack traces and health endpoint.",
            system_metrics["error_rate"],
            0.05,
            system_metrics["error_rate"] > 0.05,
        ),
    ]


def monitoring_snapshot(mode: str = "normal") -> dict[str, Any]:
    reference = load_features()
    current = make_drifted_snapshot(reference) if mode == "drifted" else reference.copy(deep=True)
    system_metrics = (
        {"p95_latency_ms": 760.0, "error_rate": 0.08}
        if mode == "drifted"
        else {"p95_latency_ms": 180.0, "error_rate": 0.0}
    )
    quality = data_quality(current)
    drift = drift_report(reference, current)
    pred = prediction_summary(current)
    alerts = build_alerts(reference, current, system_metrics)
    return {
        "mode": mode,
        "quality": quality,
        "drift": drift,
        "prediction": pred,
        "system": system_metrics,
        "alerts": [asdict(alert) for alert in alerts],
    }
