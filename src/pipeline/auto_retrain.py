"""Automated retraining gate for the CIAN price model.

This script checks data quality, freshness, route coverage, schema validity,
and simple distribution drift. If a trigger fires, it runs the Checkpoint 3
experiment runner. The goal is safe automation: decide and document first,
then retrain reproducibly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.contract_cian import validate_contract
from src.models.experiments import file_sha256

DEFAULT_CLEAN_GEO = Path("data/processed/cian_spb_clean_geo.csv")
DEFAULT_FEATURES = Path("data/features/cian_spb_offline_features.csv")
DEFAULT_METADATA = Path("data/experiments/checkpoint3_metadata.json")
DEFAULT_METRICS = Path("data/experiments/checkpoint3_metrics.csv")
DEFAULT_REPORT = Path("data/experiments/auto_retrain_decision.json")
DEFAULT_HISTORY = Path("data/experiments/auto_retrain_history.jsonl")

MAX_SNAPSHOT_AGE_DAYS = 7
MAX_MEDIAN_TARGET_DRIFT = 0.15
MAX_MEDIAN_AREA_DRIFT = 0.15
MAX_CATEGORY_SHARE_DRIFT = 0.10
MIN_GEOCODE_COVERAGE = 0.95
MIN_ROUTE_COVERAGE_OF_METRO_KNOWN = 0.80
MIN_BEST_R2 = 0.60
MAX_BEST_MAPE = 25.0


@dataclass
class Trigger:
    name: str
    fired: bool
    value: Any
    threshold: Any
    details: str


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON object or return empty dict."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def max_category_share_delta(current: pd.Series, reference: dict[str, float]) -> float:
    """Return max absolute share difference against a stored reference profile."""
    current_shares = current.astype(str).value_counts(normalize=True).to_dict()
    all_keys = set(current_shares) | set(reference)
    if not all_keys:
        return 0.0
    return max(abs(current_shares.get(key, 0.0) - reference.get(key, 0.0)) for key in all_keys)


def build_reference_profile(features: pd.DataFrame, clean_geo: pd.DataFrame) -> dict[str, Any]:
    """Build a compact profile from the current accepted training snapshot."""
    profile = {
        "median_target_price_per_sqm": float(features["target_price_per_sqm"].median()),
        "median_total_meters": float(features["total_meters"].median()),
        "rooms_count_share": features["rooms_count"].astype(str).value_counts(normalize=True).to_dict(),
        "district_share": features["district"].astype(str).value_counts(normalize=True).to_dict(),
    }
    if "lat" in clean_geo.columns and "lon" in clean_geo.columns:
        profile["geocode_coverage"] = float(clean_geo[["lat", "lon"]].notna().all(axis=1).mean())
    if {"metro_known", "distance_to_metro_route_km"}.issubset(clean_geo.columns):
        metro_known = clean_geo["metro_known"].fillna(False).astype(bool)
        profile["route_coverage_of_metro_known"] = (
            float(clean_geo.loc[metro_known, "distance_to_metro_route_km"].notna().mean())
            if metro_known.any()
            else 0.0
        )
    return profile


def relative_delta(current: float, reference: float) -> float:
    """Relative absolute delta, guarded against zero reference."""
    if reference == 0:
        return 0.0 if current == 0 else 1.0
    return abs(current - reference) / abs(reference)


def evaluate_triggers(
    clean_geo: pd.DataFrame,
    features: pd.DataFrame,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
) -> tuple[list[Trigger], dict[str, Any]]:
    """Evaluate all retraining triggers and return a fresh current profile."""
    triggers: list[Trigger] = []

    violations = validate_contract(clean_geo)
    contract_errors = [item for item in violations if item.severity == "error"]
    triggers.append(
        Trigger(
            "data_contract_errors",
            bool(contract_errors),
            len(contract_errors),
            0,
            "; ".join(f"{item.field}:{item.rule}" for item in contract_errors) or "OK",
        )
    )

    collected = pd.to_datetime(clean_geo["collected_at"], errors="coerce", utc=True)
    newest = collected.max()
    now = datetime.now(timezone.utc)
    age_days = (now - newest.to_pydatetime()).total_seconds() / 86400 if pd.notna(newest) else 10**9
    triggers.append(
        Trigger(
            "snapshot_age_days",
            age_days > MAX_SNAPSHOT_AGE_DAYS,
            round(age_days, 2),
            MAX_SNAPSHOT_AGE_DAYS,
            "freshness based on max collected_at",
        )
    )

    geocode_coverage = float(clean_geo[["lat", "lon"]].notna().all(axis=1).mean())
    triggers.append(
        Trigger(
            "geocode_coverage",
            geocode_coverage < MIN_GEOCODE_COVERAGE,
            round(geocode_coverage, 4),
            MIN_GEOCODE_COVERAGE,
            "share of rows with non-null lat/lon",
        )
    )

    metro_known = clean_geo["metro_known"].fillna(False).astype(bool)
    route_coverage = (
        float(clean_geo.loc[metro_known, "distance_to_metro_route_km"].notna().mean())
        if metro_known.any() and "distance_to_metro_route_km" in clean_geo.columns
        else 0.0
    )
    triggers.append(
        Trigger(
            "route_coverage_of_metro_known",
            route_coverage < MIN_ROUTE_COVERAGE_OF_METRO_KNOWN,
            round(route_coverage, 4),
            MIN_ROUTE_COVERAGE_OF_METRO_KNOWN,
            "share of metro-known rows with openrouteservice distance",
        )
    )

    reference_profile = metadata.get("reference_profile") or build_reference_profile(features, clean_geo)
    current_profile = build_reference_profile(features, clean_geo)

    target_drift = relative_delta(
        current_profile["median_target_price_per_sqm"],
        reference_profile["median_target_price_per_sqm"],
    )
    triggers.append(
        Trigger(
            "median_target_price_per_sqm_drift",
            target_drift > MAX_MEDIAN_TARGET_DRIFT,
            round(target_drift, 4),
            MAX_MEDIAN_TARGET_DRIFT,
            "relative median target drift",
        )
    )

    area_drift = relative_delta(
        current_profile["median_total_meters"],
        reference_profile["median_total_meters"],
    )
    triggers.append(
        Trigger(
            "median_total_meters_drift",
            area_drift > MAX_MEDIAN_AREA_DRIFT,
            round(area_drift, 4),
            MAX_MEDIAN_AREA_DRIFT,
            "relative median area drift",
        )
    )

    rooms_share_delta = max_category_share_delta(features["rooms_count"], reference_profile["rooms_count_share"])
    triggers.append(
        Trigger(
            "rooms_count_share_drift",
            rooms_share_delta > MAX_CATEGORY_SHARE_DRIFT,
            round(rooms_share_delta, 4),
            MAX_CATEGORY_SHARE_DRIFT,
            "max absolute category share delta",
        )
    )

    district_share_delta = max_category_share_delta(features["district"], reference_profile["district_share"])
    triggers.append(
        Trigger(
            "district_share_drift",
            district_share_delta > MAX_CATEGORY_SHARE_DRIFT,
            round(district_share_delta, 4),
            MAX_CATEGORY_SHARE_DRIFT,
            "max absolute category share delta",
        )
    )

    if not metrics.empty and {"test_r2_per_sqm", "test_mape_percent"}.issubset(metrics.columns):
        model_rows = metrics[metrics["model"] != "B2_non_ml_baseline"].copy()
        best_r2 = float(model_rows["test_r2_per_sqm"].max()) if not model_rows.empty else 0.0
        best_mape = float(model_rows["test_mape_percent"].min()) if not model_rows.empty else 10**9
    else:
        best_r2 = 0.0
        best_mape = 10**9

    triggers.append(
        Trigger(
            "best_model_r2_below_threshold",
            best_r2 < MIN_BEST_R2,
            round(best_r2, 4),
            MIN_BEST_R2,
            "best saved experiment R2",
        )
    )
    triggers.append(
        Trigger(
            "best_model_mape_above_threshold",
            best_mape > MAX_BEST_MAPE,
            round(best_mape, 4),
            MAX_BEST_MAPE,
            "best saved experiment MAPE",
        )
    )

    return triggers, current_profile


def run_experiments() -> None:
    """Run the experiment module in a subprocess."""
    subprocess.run([sys.executable, "-m", "src.models.experiments"], check=True)


def write_decision(report: dict[str, Any], report_path: Path, history_path: Path) -> None:
    """Persist latest decision and append history."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check retraining triggers and optionally run experiments.")
    parser.add_argument("--clean-geo", type=Path, default=DEFAULT_CLEAN_GEO)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--force", action="store_true", help="Retrain regardless of trigger state.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the decision; do not retrain.")
    args = parser.parse_args()

    clean_geo = pd.read_csv(args.clean_geo)
    features = pd.read_csv(args.features)
    metadata = load_json(args.metadata)
    metrics = pd.read_csv(args.metrics) if args.metrics.exists() else pd.DataFrame()

    triggers, current_profile = evaluate_triggers(clean_geo, features, metadata, metrics)
    fired = [trigger for trigger in triggers if trigger.fired]
    should_retrain = args.force or bool(fired)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_geo_path": str(args.clean_geo),
        "features_path": str(args.features),
        "features_sha256": file_sha256(args.features),
        "decision": "retrain" if should_retrain else "skip",
        "force": args.force,
        "dry_run": args.dry_run,
        "triggers": [asdict(trigger) for trigger in triggers],
        "fired_triggers": [trigger.name for trigger in fired],
        "current_profile": current_profile,
    }

    if should_retrain and not args.dry_run:
        run_experiments()
        report["retrain_status"] = "completed"
        report["post_retrain_metrics_path"] = str(args.metrics)
    elif should_retrain and args.dry_run:
        report["retrain_status"] = "dry_run_not_executed"
    else:
        report["retrain_status"] = "not_needed"

    write_decision(report, args.report, args.history)

    print(f"Decision: {report['decision']}")
    print(f"Fired triggers: {report['fired_triggers'] or 'none'}")
    print(f"Retrain status: {report['retrain_status']}")
    print(f"Decision report: {args.report}")


if __name__ == "__main__":
    main()
